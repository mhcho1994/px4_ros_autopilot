# Load public libraries
import time
import subprocess
import numpy
import math

# Load ROS2 related libraries
import rclpy
from rclpy.node import Node
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_inverse

# Load ROS2 messages
from std_msgs.msg import *
from geometry_msgs.msg import Quaternion, Vector3
from px4_msgs.msg import *

# Load private libraries and messages
from extgcu_msgs.msg import *
from extgcu_library import controller
from extgcu_library import get_3D_to_AzEl, get_AzEl_to_3D, get_Transform_3D, Integrate_Euler

# Define attitude commander module using Python class
class extgcu_controller_module(Node,controller):

    def __init__(self):

        # Inherit from the parent class 'Node'
        super().__init__('extgcu_controller_module')

        # Load parameters from the parent class
        super().load_parameters()

        # Define publishers
        self.attitude_setpoint_publisher_           =   self.create_publisher(
            VehicleAttitudeSetpoint,
            "fmu/vehicle_attitude_setpoint/in",
            10
        )
        self.controller_status_purblisher_          =   self.create_publisher(
            ExtGCUControllerStatus,
            'extgcu/controller_status',
            10
        )

        # Define subscribers
        self.timesync_subscriber_                   =   self.create_subscription(
            Timesync,
            'fmu/timesync/out',
            self.process_timesync,
            10
        )
        self.attitude_subscriber_                   =   self.create_subscription(
            VehicleAttitude,
            "fmu/vehicle_attitude/out",
            self.process_vehicle_attitude, 
            10
        )
        self.hover_thrust_estimate_subscriber_      =   self.create_subscription(
            HoverThrustEstimate,
            "fmu/hover_thrust_estimate/out",
            self.process_hover_thrust_estimate, 
            10
        )
        self.fm_status_subscriber_                  =   self.create_subscription(
            ExtGCUFlightManagementStatus,
            'extgcu/fm_status',
            self.process_fm_status,
            10
        )
        self.acceleration_command_subscriber_       =   self.create_subscription(
            ExtGCUAccelerationCommand,
            'extgcu/acceleration',
            self.process_acceleration_command,
            10
        )

        # Initialize parameters for timer callback function
        self.timer_period_                      =   numpy.float32(self.ctrl_dt_)
        self.timer_                             =   self.create_timer(self.timer_period_, \
                                                        self.controller_callback)
        self.temp_callback_counter_             =   numpy.uint64(0)
        self.total_callback_counter_            =   numpy.uint64(0)
        self.timestamp_checker_                 =	numpy.uint64(0)

        # Initialize parameters for controller module
        self.persistent_counter_ 	            =	numpy.uint8(0)
        self.persistent_threshold_              =   numpy.uint8(self.persistent_timer_ \
                                                        /self.timer_period_)
        self.unit_thrust_level_                 =   numpy.float32(0.0)

        # Initialize fields for attitude setpoint topic
        self.q_d_                               =   numpy.zeros(4,dtype=numpy.float32)
        self.thrust_body_                       =   numpy.zeros(3,dtype=numpy.float32)
        self.euler_d_                           =   numpy.zeros(3,dtype=numpy.float32)

        # Initialize fields for controller status topic
        self.external_controller_ready_         =   numpy.uint8(0)

        # Initialize fields of timesync topic
        self.timesync_timestamp_                =   numpy.uint64(0)

        # Initialize fields of vehicle attitude topic
        self.attitude_q_                        =   numpy.zeros(4,dtype=numpy.float32)
        self.attitude_euler_                    =   numpy.zeros(3,dtype=numpy.float32)
    
        # Initialize fields of hover thrust estimate topic
        self.thrust_estimate_hover_thrust_      =   numpy.float32(0)
        self.thrust_estimate_hover_thrust_var_  =   numpy.float32(0)

        # Initialize fields of flight management status topic
        self.external_autopilot_engage_         =   numpy.uint8(0)

        # Initialize fields of acceleration command topic
        self.acceleration_cmd_                  =   numpy.zeros(3,dtype=numpy.float32)

        # Function initial call for preventing warning
        self.timesync_subscriber_
        self.attitude_subscriber_
        self.hover_thrust_estimate_subscriber_
        self.fm_status_subscriber_
        self.acceleration_command_subscriber_

    def controller_callback(self):

        # Create a log message at Nuttshell
        self.get_logger().info('Controller Callback')

        # Persistent counter: increase counter number when on, reset counter to zero when off
        # Quadrotor switches to offboard/attitude mode when persistent counter exceeds threshold
        if self.external_autopilot_engage_ == 1:

            if self.persistent_counter_ < self.persistent_threshold_:
                self.persistent_counter_            +=  1

            else:
                self.external_controller_ready_     =   1
        
        else:
            self.persistent_counter_    =   0

        self.publish_controller_status()

        # Attitude control (command)
        if self.external_autopilot_engage_ == 1:

            total_acceleration_     =   numpy.sqrt(numpy.power(self.acceleration_cmd_[0],2) \
                                            +numpy.power(self.acceleration_cmd_[1],2) \
                                            +numpy.power(self.acceleration_cmd_[2]- \
                                            self.gravity_const,2))

            self.unit_thrust_level_ =   self.thrust_estimate_hover_thrust_/ \
                                        (self.quadrotor_mass*self.gravity_const)                    # [/N] Thrust level per unit force

            self.euler_d_[0]        =   numpy.arcsin(self.acceleration_cmd_[1]/total_acceleration_) # [deg] Roll angle
            self.euler_d_[1]        =   numpy.arcsin(-self.acceleration_cmd_[0]/ \
                                            numpy.cos(self.euler_d_[0])/total_acceleration_)        # [deg] Pitch angle
            self.euler_d_[2]        =   0.0/180.0*numpy.pi                                          # [deg] Yaw angle
            temp_q_d_               =   numpy.float32(quaternion_from_euler(self.euler_d_[2], \
                                            self.euler_d_[1],self.euler_d_[0],axes='rzyx'))         # [-] Quaternion command (x-y-z-w order)
            self.q_d_               =   temp_q_d_[[3,0,1,2]]                                        # [-] Quaternion command re-ordering (w-x-y-z order)

            self.thrust_body_[0]    =   0
            self.thrust_body_[1]    =   0
            self.thrust_body_[2]    =   -self.quadrotor_mass*total_acceleration_ \
                                            *self.unit_thrust_level_                                # [%] Body-z thrust level
            
            self.publish_attitude_setpoint()

            print('--------------- Command check panel ---------------')
            print('Thrust command [%%]: %.2f' %numpy.dot(-self.thrust_body_[2],100))
            print('Quaternion command[-]: %.3f %.3f %.3f %.3f' %(self.q_d_[0],self.q_d_[1],
                                                        self.q_d_[2],self.q_d_[3]))
            print('Roll command [deg]: %.2f' %float(self.euler_d_[0]/numpy.pi*180))
            print('Pitch command [deg]: %.2f' %float(self.euler_d_[1]/numpy.pi*180))
            print('Yaw command [deg]: %.2f' %float(self.euler_d_[2]/numpy.pi*180))
            print('')
            print('=============== State check panel ===============')
            print('Hover thrust estimate [%%]: %.2f' %numpy.dot( \
                                                    self.thrust_estimate_hover_thrust_,100))
            print('Quaternion [-]: %.3f %.3f %.3f %.3f' %(self.attitude_q_[0],self.attitude_q_[1],
                                                        self.attitude_q_[2],self.attitude_q_[3]))
            print('Roll angle [deg]: %.2f' %float(self.attitude_euler_[0]/numpy.pi*180))
            print('Pitch angle [deg]: %.2f' %float(self.attitude_euler_[1]/numpy.pi*180))
            print('Yaw angle [deg]: %.2f' %float(self.attitude_euler_[2]/numpy.pi*180))
            print('')


    # Publisher functions
    def publish_attitude_setpoint(self):
        msg                                         =   VehicleAttitudeSetpoint()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.q_d                                     =   self.q_d_
        msg.thrust_body                             =   self.thrust_body_

        self.attitude_setpoint_publisher_.publish(msg)

    def publish_controller_status(self):
        msg                                         =   ExtGCUControllerStatus()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.external_controller_ready               =   int(self.external_controller_ready_)

        self.controller_status_purblisher_.publish(msg)

    # Subscriber functions
    def process_timesync(self,msg):
        self.timesync_timestamp_                    =   numpy.uint64(msg.timestamp)

    def process_vehicle_attitude(self,msg):
        self.attitude_q_                            =   numpy.float32(msg.q[[0,1,2,3]])
        self.attitude_euler_                        =   numpy.flip(euler_from_quaternion( \
                                                            self.attitude_q_[[1,2,3,0]], \
                                                            axes='rzyx'),axis=0)

    def process_hover_thrust_estimate(self,msg):
        self.thrust_estimate_hover_thrust_          =   msg.hover_thrust
        self.thrust_estimate_hover_thrust_var_      =   msg.hover_thrust_var

    def process_fm_status(self,msg):
        self.external_autopilot_engage_             =   numpy.uint8(msg.external_autopilot_engage)

    def process_acceleration_command(self,msg):
        self.acceleration_cmd_                      =   numpy.float32(msg.acceleration)

def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create extGCU attitude control node
    extgcu_controller_module_   =	extgcu_controller_module()

    # Spin the created control node
    rclpy.spin(extgcu_controller_module_)

    # After spinning, destroy the node and shutdown rclpy library
    extgcu_controller_module_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()