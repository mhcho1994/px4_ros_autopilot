#!/usr/bin/env python3

import sys

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSPresetProfiles

from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import *
from std_msgs.msg import Int32
from Controller_lib import Controller

import linecache
import ast
import math
import time
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.euler import euler2mat, euler2quat


class Controller_module(Node):

    def __init__(self):
        super().__init__('Controller_module')

        # QoS Profile
        qos_profile = QoSProfile(
            history=QoSPresetProfiles.SENSOR_DATA.value.history,
            depth=5,
            reliability=QoSPresetProfiles.SENSOR_DATA.value.reliability,
            durability=QoSPresetProfiles.SENSOR_DATA.value.durability
            )

        # Define subscribers
        self.status_subscriber_           = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        
        self.vehicle_odometry_subscriber_ = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile)
        
        self.command_pose_subscriber_     = self.create_subscription(
            PoseStamped,
            '/command/pose', 
            self.command_pose_callback,
            10)
               
        # Define publishers
        self.offboard_mode_publisher_   = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode',
            10)

        self.actuator_motors_publisher_ = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',  
            10)
        
        self.motor_failure_publisher_ = self.create_publisher(
            Int32, '/motor_failure/motor_number', 10)        
        
        # Initialize 
        timer_period = 0.005  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        
        # Initialize odometry
        self.time_temp = np.uint64(0)
        self.pos_odo = np.zeros(3, dtype=np.float32)
        self.vel_odo = np.zeros(3, dtype=np.float32)
        self.quat_odo = np.zeros(4, dtype=np.float32)
        self.angvel_odo = np.zeros(3, dtype=np.float32)

        # Initialize reference command
        self.pos_cmd = np.zeros(3, dtype=np.float32)
        self.ori_cmd = np.zeros(4, dtype=np.float32)

        # Initialize control output
        self.throttles = np.zeros(4)

        # Inintialize UAV parameters
        self.zero_position_armed = 100
        self.input_scaling = 1000
        self.thrust_constant = 5.84e-06
        self.moment_constant = 0.06       
        self.arm_length = 0.25
        self._inertia_matrix = np.array([0.029125, 0.029125, 0.055125])  # Inertia matrix
        self._gravity = 9.81  # Gravity
        self._uav_mass = 1.725  # UAV mass

        # Initialize matrices
        self.torques_and_thrust_to_rotor_velocities_ = np.zeros((4, 4))
        self.throttles_to_normalized_torques_and_thrust_ = np.zeros((4, 4))

        # Attack parameters
        self.callback_counter                   =   int(0)
        self.motors_to_fail                     =   int(0)
        self.attack_on_off                      =   int(0)

    def cmdloop_callback(self):
        # Publish offboard control modes
        self.offboard_mode_publish_(pos_cont=False, vel_cont=False, acc_cont=False, att_cont=False, act_cont=True)
        
        wrench = np.zeros(4)
        desired_quat = np.zeros(4)
        throttles = np.zeros(4)
        pos_odo1 = np.zeros(3)
        vel_odo1 = np.zeros(3)
        quat_odo1 = np.zeros(4)
        angvel_odo1 = np.zeros(3)

        controller_ = Controller()

        # Initialize UAV parameters with placeholder values
        controller_.set_uav_parameters(self._uav_mass, self._inertia_matrix, self._gravity)
        controller_.set_control_gains(np.array([7.0, 7.0, 6.0]), np.array([6.0, 6.0, 3.0]), np.array([3.5, 3.5, 0.3]), np.array([0.5, 0.5, 0.1]))
        
        pos_odo1, vel_odo1, quat_odo1, angvel_odo1 = self.eigen_odometry_from_PX4_msg(self.pos_odo, self.vel_odo, self.quat_odo, self.angvel_odo)

        # Set odometry and trajectory point
        controller_.set_odometry(pos_odo1, vel_odo1, quat_odo1, angvel_odo1)
        controller_.set_trajectory_point(self.pos_cmd, self.ori_cmd)   

        # Calculate controller output
        wrench, desired_quat = controller_.calculate_controller_output()
        throttles = self.px4InverseSITL(wrench)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            current_time = int(Clock().now().nanoseconds / 1000)

            self.callback_counter += 1

            if self.callback_counter == 1801:
                self.motors_to_fail = int(1)                    # Randomly select one motors from 1 to 4
                self.attack_on_off = 1
                self.publish_motor_failure(self.motors_to_fail)

            elif self.callback_counter == 1900:
                self.publish_motor_failure(0)
                self.callback_counter = 0
                self.attack_on_off = 0

            # Calculate controller output
            self.actuator_motors_publish_(throttles)
            # self.attitude_setpoint_publish_(wrench[3]/1000, desired_quat)


        # self.get_logger().info(f'time: {self.time_temp}, \n \
        #                          Odometry Position: {self.pos_odo}, \n Quaternion: {self.quat_odo}, \n Velocity: {self.vel_odo}, \n \
        #                          Angular Velocity: {self.angvel_odo}, \n Desired Position: {self.pos_cmd}, \n Desired Orientation: {self.ori_cmd}, \n \
        #                          Navigation State: {self.nav_state}, \n \
        #                          Desired Quaternion: {controller_.r_R_B_W}, \n Desired Yaw: {controller_.r_yaw}')
    
    def compute_ControlAllocation_and_ActuatorEffect_matrices(self):
        kDegToRad = np.pi / 180.0
        kS = np.sin(45 * kDegToRad)
        rotor_velocities_to_torques_and_thrust = np.zeros((4, 4))
        rotor_velocities_to_torques_and_thrust = np.array([
                [-kS, kS, kS, -kS],
                [-kS, kS, -kS, kS],
                [-1, -1, 1, 1],
                [1, 1, 1, 1]
                ])
        # mixing_matrix = np.array([
        #         [-0.495384, -0.707107, -0.765306, 1.0],
        #         [0.495384, 0.707107, -1.0, 1.0],
        #         [0.495384, -0.707107, 0.765306, 1.0],
        #         [-0.495384, 0.707107, 1.0, 1.0]
        #         ])
        
        ## Hardcoded because the calculation of pesudo-inverse is not accurate
        # self.throttles_to_normalized_torques_and_thrust_ = np.array([
        #         [-0.5718, 0.4376, 0.5718, -0.4376],
        #         [-0.3536, 0.3536, -0.3536, 0.3536],
        #         [-0.2832, -0.2832, 0.2832, 0.2832],
        #         [0.2500, 0.2500, 0.2500, 0.2500]
        #         ])

        # Calculate Control allocation matrix: Wrench to Rotational velocities / k: helper matrix
        k = np.array([self.thrust_constant * self.arm_length,
                      self.thrust_constant * self.arm_length,
                      self.moment_constant * self.thrust_constant,
                      self.thrust_constant])
        
        # Element-wise multiplication
        rotor_velocities_to_torques_and_thrust = np.diag(k) @ rotor_velocities_to_torques_and_thrust
        self.torques_and_thrust_to_rotor_velocities_ = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)

    def px4InverseSITL(self, wrench):
        # Initialize vectors
        omega = np.zeros(4)
        throttles = np.zeros(4)
        normalized_torque_and_thrust = np.zeros(4)
        ones_temp = np.ones(4)
        
        array = np.array([
                [-242159.856570736, -242159.856570736, -713470.319634703, 42808.2191780822],
                [242159.856570736, 242159.856570736, -713470.319634703, 42808.2191780822],
                [242159.856570736, -242159.856570736, 713470.319634703, 42808.2191780822],
                [-242159.856570736, 242159.856570736, 713470.319634703, 42808.2191780822]
                ])

        # Control allocation: Wrench to Rotational velocities (omega)
        omega = array @ wrench
        omega = np.sqrt(np.abs(omega))  # Element-wise square root, handle negative values
        
        # CBF
        indv_forces = omega * np.abs(omega) * self.thrust_constant
        print("indv_forces")
        print(indv_forces)

        u_safe = indv_forces
        
        u_safe = np.array([u_safe[0], u_safe[1], u_safe[2], u_safe[3]], dtype=np.float32)
        # # print("u_safe")
        # # print(u_safe)

        # Calculate throttles
        omega = np.sqrt(np.abs(u_safe)/self.thrust_constant)   

        print("omega")
        print(omega)

        # Calculate hrottles from omega (rotor velocities)
        throttles = (omega - (self.zero_position_armed * ones_temp))
        throttles = throttles / self.input_scaling

        print("throttles")
        print(throttles)
        
        # Inverse Mixing: throttles to normalized torques and thrust
        # normalized_torque_and_thrust = self.throttles_to_normalized_torques_and_thrust_ @ throttles

        return throttles


    # Function for Sliding Mode Conrol Barrier Function
    def hyper_tangent(self, input_signal, gain=1.0):
        
        return np.tanh(gain * input_signal)

    # Subscribers
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def command_pose_callback(self, msg):
        # Extract position and orientation from the message
        self.pos_cmd = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ori_cmd = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    def vehicle_odometry_callback(self, msg):
        self.time_temp  = msg.timestamp
        self.pos_odo = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=np.float32)
        self.vel_odo = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=np.float32)
        self.quat_odo = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=np.float32)
        self.angvel_odo = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]], dtype=np.float32)

    # Helper functions
    def rotate_vector_from_to_ENU_NED(self, vec_in):
        # NED (X North, Y East, Z Down) & ENU (X East, Y North, Z Up)
        vec_out = np.array([vec_in[1], vec_in[0], -vec_in[2]])
        return vec_out

    def rotate_vector_from_to_FRD_FLU(self, vec_in):
        # FRD (X Forward, Y Right, Z Down) & FLU (X Forward, Y Left, Z Up)
        vec_out = np.array([vec_in[0], -vec_in[1], -vec_in[2]])
        return vec_out

    def eigen_odometry_from_PX4_msg(self, pos, vel, quat, ang_vel):
        position_W = self.rotate_vector_from_to_ENU_NED(pos)
        velocity_B = self.rotate_vector_from_to_ENU_NED(vel)
        orientation_B_W = self.rotate_quaternion_from_to_ENU_NED(quat)  # ordering (w, x, y, z)
        
        angular_velocity_B = self.rotate_vector_from_to_FRD_FLU(ang_vel)
    
        return position_W, velocity_B, orientation_B_W, angular_velocity_B

    # Publisher
    def offboard_mode_publish_(self, pos_cont, vel_cont, acc_cont, att_cont, act_cont):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = bool(pos_cont)
        msg.velocity = bool(vel_cont)
        msg.acceleration = bool(acc_cont)
        msg.attitude = bool(att_cont)
        msg.actuator = bool(act_cont)

        self.offboard_mode_publisher_.publish(msg)

    def actuator_motors_publish_(self, throttles):
        msg = ActuatorMotors()
        msg.control[0] = np.float32(throttles[0])
        msg.control[1] = np.float32(throttles[1])
        msg.control[2] = np.float32(throttles[2])
        msg.control[3] = np.float32(throttles[3])
        msg.control[4] = math.nan
        msg.control[5] = math.nan
        msg.control[6] = math.nan
        msg.control[7] = math.nan
        msg.control[8] = math.nan
        msg.control[9] = math.nan
        msg.control[10] = math.nan
        msg.control[11] = math.nan
        msg.reversible_flags = int(0)
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp

        self.actuator_motors_publisher_.publish(msg)
    
    def attitude_setpoint_publish_(self, thrust, q):
        msg = AttitudeSetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.q_d = q
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = thrust

        self.attitude_setpoint_publisher_.publish(msg)

    def publish_motor_failure(self, motor_id):
        msg = Int32()
        msg.data = motor_id
        self.motor_failure_publisher_.publish(msg)
        self.get_logger().info(f'Publishing motor failure on motor: {msg.data}')

    def rotate_quaternion_from_to_ENU_NED(self, quat_in):
        # Transform from orientation represented in ROS format to PX4 format and back.
        # quat_in_reordered = [quat_in[1], quat_in[2], quat_in[3], quat_in[0]]
    
        # NED to ENU conversion Euler angles
        euler_1 = np.array([np.pi, 0.0, np.pi/2])
        NED_ENU_Q = euler2quat(euler_1[2], euler_1[1], euler_1[0], 'szyx')
        # print("NED_ENU_Q")
        # print(NED_ENU_Q)

        # Aircraft to baselink conversion Euler angles
        euler_2 = np.array([np.pi, 0.0, 0.0])
        AIRCRAFT_BASELINK_Q = euler2quat(euler_2[2], euler_2[1], euler_2[0], 'szyx')
        # print("AIRCRAFT_BASELINK_Q")
        # print(AIRCRAFT_BASELINK_Q)

        # Perform the quaternion multiplications to achieve the desired rotation
        # Note: the multiply function from transforms3d takes quaternions in [w, x, y, z] format
        result_quat = qmult(NED_ENU_Q, quat_in)
        result_quat = qmult(result_quat, AIRCRAFT_BASELINK_Q)

        # # Convert the rotated quaternion back to w, x, y, z order before returning
        return result_quat          # ordering (w, x, y, z)
    
    def quaternion_rotation_matrix(self,Q):
        # Extract the values from Q   (w-x-y-z) #### NEED TRANSPOSE
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
     
        # First row of the rotation matrix
        r00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
     
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        r12 = 2 * (q2 * q3 - q0 * q1)
     
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3 
     
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
        return rot_matrix    

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def main():
    rclpy.init(args=None)

    Controller_module_ = Controller_module()

    rclpy.spin(Controller_module_)

    Controller_module_.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()