#!/usr/bin/env python3

__author__ = "Minhyun Cho"
__contact__ = "@purdue.edu"

# python packages and modules
import sys
import numpy as np
from functools import partial

# ros packages
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSPresetProfiles
# import linecache
# import ast
# import math
# import time
# from scipy.spatial.transform import Rotation as R
# from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
# from transforms3d.euler import euler2mat, euler2quat

# msessages
from px4_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

# own library
from offboard_mode import OffboardControlType, ControllerType
# from Controller_lib import Controller


class OffboardControlModule(Node):

    def __init__(self):
        super().__init__('px4_ros2_offboard_module')

        # set publisher and subscriber quality of service profile
        qos_profile_pub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )

        qos_profile_sub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.VOLATILE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )

        # define subscribers and publishers
        self.publishers_ = {'vehicle_command':None, 'offboard_control_mode': None, 'actuator_motors': None}
        self.subscribers_ = {'vehicle_status': None, 'vehicle_odometry': None, 'pose_command': None}

        self.subscribers_['vehicle_status'] = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_subscribe_,
            qos_profile_sub)

        self.subscribers_['vehicle_odometry'] = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_subscribe_,
            qos_profile_sub)

        # self.subscribers_['pose_command'] = self.create_subscription(
        #     PoseStamped,
        #     '/ros_autopilot/in/pose_command',
        #     self.pose_command_subscribe_,
        #     qos_profile_sub)

        self.publishers_['vehicle_command'] = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile_pub)

        self.publishers_['offboard_control_mode'] = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile_pub)

        self.publishers_['actuator_motors'] = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',
            qos_profile_pub)

        self.publishers_['trajectory_setpoint'] = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile_pub)

        # parameters for callback and initialize
        # [sec,Hz] flight management timer (offboard mode should be at least 2Hz)
        self.timer_flight_management_ = self.create_timer(0.01, self.flight_mode_manage_)
        # [sec,Hz] controller timer (adjust publishing period based on controller type)
        self.timer_controller_ = self.create_timer(0.005, self.controller_output_publish_)

        # initialize odometry
        self.last_odom_receive_ = np.uint64(0)
        self.odom_position_ = np.zeros(3, dtype=np.float32)
        self.odom_velocity_ = np.zeros(3, dtype=np.float32)
        self.odom_quaternion_ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.odom_rate_ = np.zeros(3, dtype=np.float32)

        # initialize reference command
        self.command_position_ = np.zeros(3, dtype=np.float32)
        self.command_quaternion_ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # initialize controller mode, navigation state, guidance flow
        self.nav_state_ = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state_ = VehicleStatus.ARMING_STATE_DISARMED 
        self.control_mode = OffboardControlType.ACTUATOR
        self.controller_type = ControllerType.SE3_GEOMETRIC
        self.entry_excute_ = False
        self.flight_phase_ = np.uint8(0)
        self.next_phase_flag_ = False
        self.target_takeoff_alt_ = np.float32(10.0) # [m] target takeoff altitude, TODO: parameterize this value
        self.offboard_time_ = np.uint32(10)*10**6 # [us] offboard time, TODO: parameterize this value


    # callbacks
    def vehicle_odometry_subscribe_(self, msg):
        self.last_odom_receive_ = msg.timestamp
        self.odom_position_ = np.array(msg.position, dtype=np.float32)
        self.odom_velocity_ = np.array(msg.velocity, dtype=np.float32)
        self.odom_quaternion_ = np.array(msg.q, dtype=np.float32)
        self.odom_rate_ = np.array(msg.angular_velocity, dtype=np.float32)

    def vehicle_status_subscribe_(self, msg):
        self.nav_state_ = msg.nav_state
        self.arming_state_ = msg.arming_state

    # def pose_command_subscribe_(self, msg):
    #     self.command_position = msg.point
    #     self.command_quaternion = msg.quaternion

    def vehicle_command_publish_(self, command, param1=0.0, param2=0.0, param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds/1000) # [us] time after initiating node
        msg.param1 = param1 # [-] 1 for using px4 custom mode
        msg.param2 = param2 # [-] px4 custom mode flag
        msg.param3 = param3 # [-] px4 custom sub mode flag (auto)
        msg.param4 = param4 # [-] defined by MAVLink uint16 VEHICLE_CMD enum
        msg.param5 = param5 # [-] defined by MAVLink uint16 VEHICLE_CMD enum
        msg.param6 = param6 # [-] defined by MAVLink uint16 VEHICLE_CMD enum
        msg.param7 = param7 # [-] defined by MAVLink uint16 VEHICLE_CMD enum
        msg.command = command #[-] command ID
        msg.target_system = 0 # [-] system which should execute the command
        msg.target_component = 1 # [-] component which should execute the command, 0 for all components
        msg.source_system = 1 # [-] system sending the command
        msg.source_component = 1 # [-] component sending the command
        msg.from_external = True # [-] indacation of command from external enviornment
        self.publishers_['vehicle_command'].publish(msg)

    def offboard_mode_publish_(self):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds/1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False

        match self.control_mode:
            case OffboardControlType.POSITION:
                msg.position = True
            case OffboardControlType.VELOCITY:
                msg.velocity = True
            case OffboardControlType.ACCELERATION:
                msg.acceleration = True
            case OffboardControlType.ATTITUDE:
                msg.attitude = True
            case OffboardControlType.RATE:
                msg.body_rate = True
            case OffboardControlType.TORQUETHRUST:
                msg.thrust_and_torque = True
            case OffboardControlType.ACTUATOR:
                msg.direct_actuator = True
            case _:
                self.get_logger().info('unknown control mode, disable all offboard modes')

        self.publishers_['offboard_control_mode'].publish(msg)

    def actuator_motors_publish_(self, throttles):
        msg = ActuatorMotors()
        msg.timestamp = int(Clock().now().nanoseconds/1000)
        msg.timestamp_sample = msg.timestamp
        msg.reversible_flags = int(0)
        for idx in range(12):
            if idx < throttles.size:
                msg.control[idx] = np.float32(throttles[idx])
            else:
                msg.control[idx] = np.nan

        self.publishers_['actuator_motors'].publish(msg)

    def trajectory_sepoint_publish_(self, setpoint, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds/1000)
        msg.position = np.array(setpoint, dtype=np.float32)
        msg.yaw = float(yaw)

        self.publishers_['trajectory_setpoint'].publish(msg)

    def flight_mode_manage_(self):
        # flight phase 0: idle/arming/takeoff
        if self.flight_phase_ == 0:
            # entry:
            if self.entry_excute_ == False:
                self.get_logger().info('flight phase 0: idle/arming/auto-takeoff')
                self.entry_excute_ = True
            # during:
            # self.vehicle_command_publish_(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 2.0) # [-] default auto-takeoff to 2.5m ASL
            self.vehicle_command_publish_(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, float(self.target_takeoff_alt_)) # [deg, m] yaw in NED, lat/LON in WGS-84, altitude AMSL
            self.vehicle_command_publish_(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            # exit:
            if self.odom_position_[2] < -self.target_takeoff_alt_+1.0:
                self.get_logger().info('reaching preset altitude, switching to next flight phase')
                self.next_phase_flag_ = True

        elif self.flight_phase_ == 1:
            # entry:
            if self.entry_execute_ == False:
                self.get_logger().info('flight phase 1: offboard control')
                self.offboard_mode_publish_()
                self.offboard_clock_ = int(Clock().now().nanoseconds/1000)
                self.entry_execute_ = True
            # during:
            if self.nav_state_ != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.vehicle_command_publish_(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0) # [-] switch to offboard mode
            self.offboard_mode_publish_()
            # exit:
            if int(Clock().now().nanoseconds/1000)-self.offboard_clock_ > self.offboard_time_:
                self.get_logger().info('offboard time is over, switching to next flight phase')
                self.next_phase_flag_ = True

        elif self.flight_phase_ == 2:
            # entry:
            if self.entry_execute_ is False:
                self.get_logger().info('flight phase 2: landing')
                self.entry_execute_ = True
            # during:
            if self.nav_state_ != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.vehicle_command_publish_(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
            # exit:
            if self.arming_state_ == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info('landing is done, switching to next flight phase')
                self.next_phase_flag_ = True

        else:
            if self.entry_execute_ is False:
                self.get_logger().info('flight finished')
                self.entry_execute_ = True

        if self.next_phase_flag_:
            past_flag_temp_ = self.flight_phase_
            self.flight_phase_ = self.flight_phase_+1
            new_flag_temp_ = self.flight_phase_
            self.next_phase_flag_ = False
            self.entry_execute_ = False

            self.get_logger().info('next flight phase %d -> %d' %(past_flag_temp_,new_flag_temp_))

    def controller_output_publish_(self):
        if self.nav_state_ == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            match self.control_mode:
                case OffboardControlType.POSITION:
                    self.trajectory_sepoint_publish_(np.array([0.0, 0.0, -20.0], dtype=np.float32), np.float32(0.0))
                case OffboardControlType.VELOCITY:
                    self.get_logger().info('not supported yet: choose POSITION or ACTUATOR control mode')
                case OffboardControlType.ACCELERATION:
                    self.get_logger().info('not supported yet: choose POSITION or ACTUATOR control mode')
                case OffboardControlType.ATTITUDE:
                    self.get_logger().info('not supported yet: choose POSITION or ACTUATOR control mode')
                case OffboardControlType.RATE:
                    self.get_logger().info('not supported yet: choose POSITION or ACTUATOR control mode')
                case OffboardControlType.TORQUETHRUST:
                    self.get_logger().info('not supported yet: choose POSITION or ACTUATOR control mode')
                case OffboardControlType.ACTUATOR:
                    self.actuator_motors_publish_(np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32))
                case _:
                    self.get_logger().info('unknown control mode, disable all offboard modes')

    def control_allocation(self, desired_wrench):
        self.

# INFO  [control_allocator]   Effectiveness.T =
#   | 0      | 1      | 2      | 3      | 4      | 5      
#  0|-1.43000  0.84500  0.32500  0        0       -6.50000 
#  1| 1.30000 -0.84500  0.32500  0        0       -6.50000 
#  2| 1.43000  0.84500 -0.32500  0        0       -6.50000 
#  3|-1.30000 -0.84500 -0.32500  0        0       -6.50000 
#  4| 0        0        0        0        0        0       
#  5| 0        0        0        0        0        0       
#  6| 0        0        0        0        0        0       
#  7| 0        0        0        0        0        0       
#  8| 0        0        0        0        0        0       
#  9| 0        0        0        0        0        0       
# 10| 0        0        0        0        0        0       
# 11| 0        0        0        0        0        0       
# 12| 0        0        0        0        0        0       
# 13| 0        0        0        0        0        0       
# 14| 0        0        0        0        0        0       
# 15| 0        0        0        0        0        0       
# INFO  [control_allocator]   minimum =
#   | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |10      |11      |12      |13      |14      |15      
#  0| 0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0       
# INFO  [control_allocator]   maximum =
#   | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      |10      |11      |12      |13      |14      |15      
#  0| 1.00000  1.00000  1.00000  1.00000  0        0        0        0        0        0        0        0        0        0        0        0       
# INFO  [control_allocator]   Configured actuators: 4
# control_allocator: cycle: 8969 events, 0us elapsed, 0.00us avg, min 0us max 0us 0.000us rms
#       


        return np.clip(desired_wrench, 0.0, 1.0)  # Ensure throttles are within [0, 1] range

def main():

    rclpy.init(args=None)

    offboard_control_module_ = OffboardControlModule()

    rclpy.spin(offboard_control_module_)

    offboard_control_module_.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()




        # # initialze reference command
        # self.

        # # initialze
        # self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # # Initialize control output
        # self.throttles = np.zeros(4)

        # # Inintialize UAV parameters
        # self.zero_position_armed = 100
        # self.input_scaling = 1000
        # self.thrust_constant = 5.84e-06
        # self.moment_constant = 0.06
        # self.arm_length = 0.25
        # self._inertia_matrix = np.array([0.029125, 0.029125, 0.055125])  # Inertia matrix
        # self._gravity = 9.81  # Gravity
        # self._uav_mass = 1.725  # UAV mass

        # # Initialize matrices
        # self.torques_and_thrust_to_rotor_velocities_ = np.zeros((4, 4))
        # self.throttles_to_normalized_torques_and_thrust_ = np.zeros((4, 4))

        # # Attack parameters
        # self.callback_counter                   =   int(0)
        # self.motors_to_fail                     =   int(0)
        # self.attack_on_off                      =   int(0)

    # def cmdloop_callback(self):
    #     # Publish offboard control modes
    #     self.offboard_mode_publish_(pos_cont=False, vel_cont=False, acc_cont=False, att_cont=False, act_cont=True)

    #     wrench = np.zeros(4)
    #     desired_quat = np.zeros(4)
    #     throttles = np.zeros(4)
    #     pos_odo1 = np.zeros(3)
    #     vel_odo1 = np.zeros(3)
    #     quat_odo1 = np.zeros(4)
    #     angvel_odo1 = np.zeros(3)

    #     controller_ = Controller()

    #     # Initialize UAV parameters with placeholder values
    #     controller_.set_uav_parameters(self._uav_mass, self._inertia_matrix, self._gravity)
    #     controller_.set_control_gains(np.array([7.0, 7.0, 6.0]), np.array([6.0, 6.0, 3.0]), np.array([3.5, 3.5, 0.3]), np.array([0.5, 0.5, 0.1]))

    #     pos_odo1, vel_odo1, quat_odo1, angvel_odo1 = self.eigen_odometry_from_PX4_msg(self.pos_odo, self.vel_odo, self.quat_odo, self.angvel_odo)

    #     # Set odometry and trajectory point
    #     controller_.set_odometry(pos_odo1, vel_odo1, quat_odo1, angvel_odo1)
    #     controller_.set_trajectory_point(self.pos_cmd, self.ori_cmd)

    #     # Calculate controller output
    #     wrench, desired_quat = controller_.calculate_controller_output()
    #     throttles = self.px4InverseSITL(wrench)

    #     if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
    #         current_time = int(Clock().now().nanoseconds / 1000)

    #         self.callback_counter += 1

    #         if self.callback_counter == 1801:
    #             self.motors_to_fail = int(1)                    # Randomly select one motors from 1 to 4
    #             self.attack_on_off = 1
    #             self.publish_motor_failure(self.motors_to_fail)

    #         elif self.callback_counter == 1900:
    #             self.publish_motor_failure(0)
    #             self.callback_counter = 0
    #             self.attack_on_off = 0

    #         # Calculate controller output
    #         self.actuator_motors_publish_(throttles)
    #         # self.attitude_setpoint_publish_(wrench[3]/1000, desired_quat)

    #     # self.get_logger().info(f'time: {self.time_temp}, \n \
    #     #                          Odometry Position: {self.pos_odo}, \n Quaternion: {self.quat_odo}, \n Velocity: {self.vel_odo}, \n \
    #     #                          Angular Velocity: {self.angvel_odo}, \n Desired Position: {self.pos_cmd}, \n Desired Orientation: {self.ori_cmd}, \n \
    #     #                          Navigation State: {self.nav_state}, \n \
    #     #                          Desired Quaternion: {controller_.r_R_B_W}, \n Desired Yaw: {controller_.r_yaw}')

    # def compute_ControlAllocation_and_ActuatorEffect_matrices(self):
    #     kDegToRad = np.pi / 180.0
    #     kS = np.sin(45 * kDegToRad)
    #     rotor_velocities_to_torques_and_thrust = np.zeros((4, 4))
    #     rotor_velocities_to_torques_and_thrust = np.array([
    #             [-kS, kS, kS, -kS],
    #             [-kS, kS, -kS, kS],
    #             [-1, -1, 1, 1],
    #             [1, 1, 1, 1]
    #             ])
    #     # mixing_matrix = np.array([
    #     #         [-0.495384, -0.707107, -0.765306, 1.0],
    #     #         [0.495384, 0.707107, -1.0, 1.0],
    #     #         [0.495384, -0.707107, 0.765306, 1.0],
    #     #         [-0.495384, 0.707107, 1.0, 1.0]
    #     #         ])

    #     ## Hardcoded because the calculation of pesudo-inverse is not accurate
    #     # self.throttles_to_normalized_torques_and_thrust_ = np.array([
    #     #         [-0.5718, 0.4376, 0.5718, -0.4376],
    #     #         [-0.3536, 0.3536, -0.3536, 0.3536],
    #     #         [-0.2832, -0.2832, 0.2832, 0.2832],
    #     #         [0.2500, 0.2500, 0.2500, 0.2500]
    #     #         ])

    #     # Calculate Control allocation matrix: Wrench to Rotational velocities / k: helper matrix
    #     k = np.array([self.thrust_constant * self.arm_length,
    #                   self.thrust_constant * self.arm_length,
    #                   self.moment_constant * self.thrust_constant,
    #                   self.thrust_constant])

    #     # Element-wise multiplication
    #     rotor_velocities_to_torques_and_thrust = np.diag(k) @ rotor_velocities_to_torques_and_thrust
    #     self.torques_and_thrust_to_rotor_velocities_ = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)

    # def px4InverseSITL(self, wrench):
    #     # Initialize vectors
    #     omega = np.zeros(4)
    #     throttles = np.zeros(4)
    #     normalized_torque_and_thrust = np.zeros(4)
    #     ones_temp = np.ones(4)

    #     array = np.array([
    #             [-242159.856570736, -242159.856570736, -713470.319634703, 42808.2191780822],
    #             [242159.856570736, 242159.856570736, -713470.319634703, 42808.2191780822],
    #             [242159.856570736, -242159.856570736, 713470.319634703, 42808.2191780822],
    #             [-242159.856570736, 242159.856570736, 713470.319634703, 42808.2191780822]
    #             ])

    #     # Control allocation: Wrench to Rotational velocities (omega)
    #     omega = array @ wrench
    #     omega = np.sqrt(np.abs(omega))  # Element-wise square root, handle negative values

    #     # CBF
    #     indv_forces = omega * np.abs(omega) * self.thrust_constant
    #     print("indv_forces")
    #     print(indv_forces)

    #     u_safe = indv_forces

    #     u_safe = np.array([u_safe[0], u_safe[1], u_safe[2], u_safe[3]], dtype=np.float32)
    #     # # print("u_safe")
    #     # # print(u_safe)

    #     # Calculate throttles
    #     omega = np.sqrt(np.abs(u_safe)/self.thrust_constant)

    #     print("omega")
    #     print(omega)

    #     # Calculate hrottles from omega (rotor velocities)
    #     throttles = (omega - (self.zero_position_armed * ones_temp))
    #     throttles = throttles / self.input_scaling

    #     print("throttles")
    #     print(throttles)

    #     # Inverse Mixing: throttles to normalized torques and thrust
    #     # normalized_torque_and_thrust = self.throttles_to_normalized_torques_and_thrust_ @ throttles

    #     return throttles

    # # Function for Sliding Mode Conrol Barrier Function
    # def hyper_tangent(self, input_signal, gain=1.0):

    #     return np.tanh(gain * input_signal)

    # # Subscribers
    # def vehicle_status_callback(self, msg):
    #     # TODO: handle NED->ENU transformation
    #     print("NAV_STATUS: ", msg.nav_state)
    #     print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
    #     self.nav_state = msg.nav_state

    # def command_pose_callback(self, msg):
    #     # Extract position and orientation from the message
    #     self.pos_cmd = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #     self.ori_cmd = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    # # Helper functions

    # def rotate_vector_from_to_ENU_NED(self, vec_in):
    #     # NED (X North, Y East, Z Down) & ENU (X East, Y North, Z Up)
    #     vec_out = np.array([vec_in[1], vec_in[0], -vec_in[2]])
    #     return vec_out

    # def rotate_vector_from_to_FRD_FLU(self, vec_in):
    #     # FRD (X Forward, Y Right, Z Down) & FLU (X Forward, Y Left, Z Up)
    #     vec_out = np.array([vec_in[0], -vec_in[1], -vec_in[2]])
    #     return vec_out

    # def eigen_odometry_from_PX4_msg(self, pos, vel, quat, ang_vel):
    #     position_W = self.rotate_vector_from_to_ENU_NED(pos)
    #     velocity_B = self.rotate_vector_from_to_ENU_NED(vel)
    #     orientation_B_W = self.rotate_quaternion_from_to_ENU_NED(
    #         quat)  # ordering (w, x, y, z)

    #     angular_velocity_B = self.rotate_vector_from_to_FRD_FLU(ang_vel)

    #     return position_W, velocity_B, orientation_B_W, angular_velocity_B

    # Publisher



    # def attitude_setpoint_publish_(self, thrust, q):
    #     msg = AttitudeSetpoint()
    #     msg.timestamp = int(Clock().now().nanoseconds / 1000)
    #     msg.q_d = q
    #     msg.thrust_body[0] = 0.0
    #     msg.thrust_body[1] = 0.0
    #     msg.thrust_body[2] = thrust

    #     self.attitude_setpoint_publisher_.publish(msg)

    # def publish_motor_failure(self, motor_id):
    #     msg = Int32()
    #     msg.data = motor_id
    #     self.motor_failure_publisher_.publish(msg)
    #     self.get_logger().info(
    #         f'Publishing motor failure on motor: {msg.data}')

    # def rotate_quaternion_from_to_ENU_NED(self, quat_in):
    #     # Transform from orientation represented in ROS format to PX4 format and back.
    #     # quat_in_reordered = [quat_in[1], quat_in[2], quat_in[3], quat_in[0]]

    #     # NED to ENU conversion Euler angles
    #     euler_1 = np.array([np.pi, 0.0, np.pi/2])
    #     NED_ENU_Q = euler2quat(euler_1[2], euler_1[1], euler_1[0], 'szyx')
    #     # print("NED_ENU_Q")
    #     # print(NED_ENU_Q)

    #     # Aircraft to baselink conversion Euler angles
    #     euler_2 = np.array([np.pi, 0.0, 0.0])
    #     AIRCRAFT_BASELINK_Q = euler2quat(
    #         euler_2[2], euler_2[1], euler_2[0], 'szyx')
    #     # print("AIRCRAFT_BASELINK_Q")
    #     # print(AIRCRAFT_BASELINK_Q)

    #     # Perform the quaternion multiplications to achieve the desired rotation
    #     # Note: the multiply function from transforms3d takes quaternions in [w, x, y, z] format
    #     result_quat = qmult(NED_ENU_Q, quat_in)
    #     result_quat = qmult(result_quat, AIRCRAFT_BASELINK_Q)

    #     # # Convert the rotated quaternion back to w, x, y, z order before returning
    #     return result_quat          # ordering (w, x, y, z)

    # def quaternion_rotation_matrix(self, Q):
    #     # Extract the values from Q   (w-x-y-z) #### NEED TRANSPOSE
    #     q0 = Q[0]
    #     q1 = Q[1]
    #     q2 = Q[2]
    #     q3 = Q[3]

    #     # First row of the rotation matrix
    #     r00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    #     r01 = 2 * (q1 * q2 - q0 * q3)
    #     r02 = 2 * (q1 * q3 + q0 * q2)

    #     # Second row of the rotation matrix
    #     r10 = 2 * (q1 * q2 + q0 * q3)
    #     r11 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    #     r12 = 2 * (q2 * q3 - q0 * q1)

    #     # Third row of the rotation matrix
    #     r20 = 2 * (q1 * q3 - q0 * q2)
    #     r21 = 2 * (q2 * q3 + q0 * q1)
    #     r22 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    #     # 3x3 rotation matrix
    #     rot_matrix = np.array([[r00, r01, r02],
    #                            [r10, r11, r12],
    #                            [r20, r21, r22]])

    #     return rot_matrix

    # def euler_from_quaternion(self, x, y, z, w):
    #     t0 = +2.0 * (w * x + y * z)
    #     t1 = +1.0 - 2.0 * (x * x + y * y)
    #     roll_x = math.atan2(t0, t1)

    #     t2 = +2.0 * (w * y - z * x)
    #     t2 = +1.0 if t2 > +1.0 else t2
    #     t2 = -1.0 if t2 < -1.0 else t2
    #     pitch_y = math.asin(t2)

    #     t3 = +2.0 * (w * z + x * y)
    #     t4 = +1.0 - 2.0 * (y * y + z * z)
    #     yaw_z = math.atan2(t3, t4)

    #     return roll_x, pitch_y, yaw_z  # in radians



