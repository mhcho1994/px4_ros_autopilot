# Load public libraries
import time
import subprocess
import numpy

# Load ROS2 related libraries
import rclpy
from rclpy.node import Node
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_inverse

# Load ROS2 messages
from std_msgs.msg import *
from geometry_msgs.msg import Quaternion, Vector3
from px4_msgs.msg import *
from extgcu_msgs.msg import *

# Define flight management module using Python class
class extgcu_fm_module(Node):

    def __init__(self):

        # Inherit from the parent class 'Node'
        super().__init__('extgcu_fm_module')

        # Define publishers
        self.vehicle_command_publisher_             =   self.create_publisher(
            VehicleCommand,
            'fmu/vehicle_command/in',
            10
        )
        self.trajectory_setpoint_publisher_         =   self.create_publisher(
            TrajectorySetpoint,
            'fmu/trajectory_setpoint/in',
            10
        )
        self.offboard_control_mode_publisher_       =   self.create_publisher(
            OffboardControlMode,
            'fmu/offboard_control_mode/in',
            10
        )
        self.fm_status_publisher_                   =   self.create_publisher(
            ExtGCUFlightManagementStatus,
            'extgcu/fm_status',
            10
        )

        # Define subscribers
        self.timesync_subscriber_                   =   self.create_subscription(
            Timesync,
            'fmu/timesync/out',
            self.process_timesync,
            10
        )
        self.vehicle_status_subscriber_             =   self.create_subscription(
            VehicleStatus,
            'fmu/vehicle_status/out',
            self.process_vehicle_status, 
            10
        )
        self.takeoff_status_subscriber_             =   self.create_subscription(
            TakeoffStatus,
            'fmu/takeoff_status/out',
            self.process_takeoff_status,
            10
        )
        self.vehicle_local_position_subscriber_     =   self.create_subscription(
            VehicleLocalPosition,
            'fmu/vehicle_local_position/out',
            self.process_vehicle_local_position,
            10
        )
        self.autopilot_status_subscriber_           =   self.create_subscription(
            ExtGCUAutopilotStatus,
            'extgcu/autopilot_status',
            self.process_autopilot_status,
            10
        )
        self.controller_status_subscriber_          =   self.create_subscription(
            ExtGCUControllerStatus,
            'extgcu/controller_status',
            self.process_controller_status,
            10
        )
        self.vehicle_trajectory_subscriber_         =   self.create_subscription(
            ExtGCUTrajectory,
            'extgcu/trajectory',
            self.process_vehicle_trajectory,
            10
        )

        # Initialize parameters for timer callback function
        self.timer_period                           =   numpy.float32(0.1)
        self.timer                                  =   self.create_timer(self.timer_period, \
                                                            self.fm_callback)
        self.temp_callback_counter_                 =   numpy.uint64(0)
        self.total_callback_counter_                =   numpy.uint64(0)
        self.timestamp_checker_                     =	numpy.uint64(0)

        # Initialize parameters for flight management module
        self.flight_phase_ 	                        =	numpy.uint8(0)
        self.entry_execute_                         =	numpy.uint8(1)
        self.nav_xy_accept_rad_                     =   numpy.float32(2.0)
        self.nav_z_accept_rad_                      =   numpy.float32(0.8)
        self.nav_vz_accept_bound_                   =   numpy.float32(1.5)

        # Initialize fields for vehicle command topic
        self.PX4_CUSTOM_MAIN_MODE_MANUAL_           =   numpy.uint8(1)
        self.PX4_CUSTOM_MAIN_MODE_ALTCTL_           =   numpy.uint8(2)
        self.PX4_CUSTOM_MAIN_MODE_POSCTL_           =   numpy.uint8(3)
        self.PX4_CUSTOM_MAIN_MODE_AUTO_             =   numpy.uint8(4)
        self.PX4_CUSTOM_MAIN_MODE_ACRO_             =   numpy.uint8(5)
        self.PX4_CUSTOM_MAIN_MODE_OFFBOARD_         =   numpy.uint8(6)
        self.PX4_CUSTOM_MAIN_MODE_STABILIZED_       =   numpy.uint8(7)

        self.PX4_CUSTOM_SUB_MODE_AUTO_READY_        =   numpy.uint8(1) 						        # Debug: Warning when engaged (Unsupported auto mode)
        self.PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF_      =   numpy.uint8(2)
        self.PX4_CUSTOM_SUB_MODE_AUTO_LOITER_       =   numpy.uint8(3)
        self.PX4_CUSTOM_SUB_MODE_AUTO_MISSION_      =   numpy.uint8(4)
        self.PX4_CUSTOM_SUB_MODE_AUTO_RTL_          =   numpy.uint8(5)
        self.PX4_CUSTOM_SUB_MODE_AUTO_LAND_         =   numpy.uint8(6)
        self.PX4_CUSTOM_SUB_MODE_AUTO_FOLLOW_TARGET_=   numpy.uint8(8)
        self.PX4_CUSTOM_SUB_MODE_AUTO_PRECLAND_     =   numpy.uint8(9)
        self.PX4_CUSTOM_SUB_MODE_AUTO_VTOL_TAKEOFF_ =   numpy.uint8(10)

        self.ARMING_ACTION_DISARM_ 					=	numpy.uint8(0)
        self.ARMING_ACTION_ARM_ 					=	numpy.uint8(1)

        # Initialize fields for trajectory setpoint topic
        self.trajectory_setpoint_x_ 				=	numpy.float32(0.0)
        self.trajectory_setpoint_y_ 				=	numpy.float32(0.0)
        self.trajectory_setpoint_z_ 				=	numpy.float32(0.0)
        self.trajectory_setpoint_yaw_ 				=	numpy.float32(0.0)

        # Initialize fields of offboard control mode topic
        self.offboard_ctrl_position_                =	False
        self.offboard_ctrl_velocity_                =	False
        self.offboard_ctrl_acceleration_            =	False
        self.offboard_ctrl_attitude_                =	False
        self.offboard_ctrl_body_rate_	            =	False
        self.offboard_ctrl_actuator_                =	False

        # Initialize fields of flight management status topic
        self.external_autopilot_engage_             =   numpy.uint8(0)

        # Initialize fields of timesync topic
        self.timesync_timestamp_              	    =   numpy.uint64(0)

        # Initialize fields of vehicle status topic
        self.status_nav_state_                      =   numpy.uint8(0)
        self.status_arming_status_                  =   numpy.uint8(0)

        self.NAVIGATION_STATE_MANUAL_ 				= 	numpy.uint8(0)
        self.NAVIGATION_STATE_ALTCTL_ 				= 	numpy.uint8(1)
        self.NAVIGATION_STATE_POSCTL_ 				= 	numpy.uint8(2)
        self.NAVIGATION_STATE_AUTO_MISSION_ 		= 	numpy.uint8(3)
        self.NAVIGATION_STATE_AUTO_LOITER_ 			= 	numpy.uint8(4)
        self.NAVIGATION_STATE_AUTO_RTL_ 			= 	numpy.uint8(5)
        self.NAVIGATION_STATE_AUTO_LANDENGFAIL_ 	= 	numpy.uint8(8)
        self.NAVIGATION_STATE_ACRO_ 				=	numpy.uint8(10)
        self.NAVIGATION_STATE_DESCEND_ 				= 	numpy.uint8(12)
        self.NAVIGATION_STATE_TERMINATION_ 			= 	numpy.uint8(13)
        self.NAVIGATION_STATE_OFFBOARD_ 			= 	numpy.uint8(14)
        self.NAVIGATION_STATE_STAB_ 				= 	numpy.uint8(15)
        self.NAVIGATION_STATE_AUTO_TAKEOFF_ 		= 	numpy.uint8(17)
        self.NAVIGATION_STATE_AUTO_LAND_ 			= 	numpy.uint8(18)
        self.NAVIGATION_STATE_AUTO_FOLLOW_TARGET_ 	= 	numpy.uint8(19)
        self.NAVIGATION_STATE_AUTO_PRECLAND_ 		= 	numpy.uint8(20)
        self.NAVIGATION_STATE_ORBIT_ 				= 	numpy.uint8(21)
        self.NAVIGATION_STATE_AUTO_VTOL_TAKEOFF_ 	= 	numpy.uint8(22)

        self.ARMING_STATE_INIT_                     =   numpy.uint8(0)
        self.ARMING_STATE_STANDBY_                  =   numpy.uint8(1)
        self.ARMING_STATE_ARMED_                    =   numpy.uint8(2)

        # Initialize fields of takeoff status topic
        self.status_takeoff_                        =   numpy.uint8(0)

        self.TAKEOFF_STATE_UNINITIALIZED_     		= 	numpy.uint8(0)
        self.TAKEOFF_STATE_DISARMED_          		= 	numpy.uint8(1)
        self.TAKEOFF_STATE_SPOOLUP_           		= 	numpy.uint8(2)
        self.TAKEOFF_STATE_READY_FOR_TAKEOFF_       =   numpy.uint8(3)
        self.TAKEOFF_STATE_RAMPUP_                  =   numpy.uint8(4)
        self.TAKEOFF_STATE_FLIGHT_                  =   numpy.uint8(5)

        # Initialize fields of local position topic
        self.vehicle_local_position_x_              =   numpy.float32(0.0)
        self.vehicle_local_position_y_              =   numpy.float32(0.0)
        self.vehicle_local_position_z_              =   numpy.float32(0.0)
        self.vehicle_local_position_vx_             =   numpy.float32(0.0)
        self.vehicle_local_position_vy_             =   numpy.float32(0.0)
        self.vehicle_local_position_vz_             =   numpy.float32(0.0)

        # Initialize fields of autopilot status topic
        self.external_autopilot_ready_              =   numpy.uint8(0)

        # Initialize fields of controller status topic
        self.external_controller_ready_             =   numpy.uint8(0)

        # Initialize fields of 
        self.wpts_targeted_offboard_             =   numpy.zeros((3,3),dtype=numpy.float32)

        # Declare subscribers once to avoid unused variable warning
        self.timesync_subscriber_
        self.vehicle_status_subscriber_
        self.takeoff_status_subscriber_
        self.vehicle_local_position_subscriber_
        self.autopilot_status_subscriber_ 
        self.controller_status_subscriber_
        self.vehicle_trajectory_subscriber_

    # Flight management module callback function
    def fm_callback(self):

        # ------------------- Offboard control test code -------------------
        # if (self.total_callback_counter_ > numpy.uint64(10/self.timer_period)) \
        #     and (self.total_callback_counter_ <= numpy.uint64(15/self.timer_period)):

        #     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, \
        #         param1=self.ARMING_ACTION_ARM_)
        #     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
        #         param2=self.PX4_CUSTOM_MAIN_MODE_AUTO_,param3=self.PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF_)

        # elif (self.total_callback_counter_ > numpy.uint64(15/self.timer_period)):
        #     self.publish_trajectory_setpoint()
        #     self.offboard_ctrl_position_  = True
        #     self.publish_offboard_control_mode()
        #     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1,\
        #         param2=self.PX4_CUSTOM_MAIN_MODE_OFFBOARD_)

        # else:
        #     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
        #         param2=self.PX4_CUSTOM_MAIN_MODE_MANUAL_)

        # ------------------- Flight simulation using external autopilot -------------------
        # Create a log message at NuttShell
        self.get_logger().info('Flight Manager Callback')
        self.publish_fm_status()

        # This code is for testing the developed ROS2-based guidance algorithm
        # It need to be revised when the code is built and integrated in the real hardware

        # Phase 0: Manual(idle) - Switch to manual mode (From any mode of last missions)
        # Set PX4_CUSTOM_MAIN_MODE to PX4_CUSTOM_MAIN_MODE_MANUAL and hold this mode for 10 seconds
        if self.flight_phase_ == 0:
            
            # entry:
            if self.entry_execute_:

                self.entry_execute_ 			= 	numpy.uint64(0)
                self.temp_callback_counter_		=   numpy.uint64(0)
                self.total_lapse_counter_  		=   numpy.uint64(0)
            
            # during:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
                param2=self.PX4_CUSTOM_MAIN_MODE_MANUAL_)
            self.get_logger().info("Current Mode: Manual (Idle)")

            # transition:
            if (self.status_nav_state_ == self.NAVIGATION_STATE_MANUAL_) \
                and (self.temp_callback_counter_ > numpy.uint64(10/self.timer_period)):

                # exit:
                self.flight_phase_ 			=	1
                self.entry_execute_ 		=	1

        # Phase 1: AutoTakeOff - Switch to auto/takeoff mode
        # Arm the multicopter by setting VEHICLE_CMD_COMPONENT_ARM_DISARM to ARMING_ACTION_ARM
        # Engage auto mode and takeoff submode
        if self.flight_phase_ == 1:
            
            # entry:
            if self.entry_execute_:

                self.entry_execute_ 			= 	numpy.uint64(0)
                self.temp_callback_counter_		=   numpy.uint64(0)

            # during:
            if (self.status_arming_status_ == self.ARMING_STATE_STANDBY_) \
                and (self.status_takeoff_ <= self.TAKEOFF_STATE_READY_FOR_TAKEOFF_):

                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, \
                    self.ARMING_ACTION_ARM_)
                self.get_logger().info("Current Mode: TakeOff (Arm)")

            elif (self.status_arming_status_ == self.ARMING_STATE_ARMED_):

                # Debug: status_take directly becomes 5 from 0 when arming is engaged
                # Need to be checked whether this originates from the low communication rate between
                # ROS2 - PX4 or inherent setting of this module (or might be bug)
                # \ and (self.status_takeoff_ <= self.TAKEOFF_STATE_FLIGHT_): -> removed
                # This might also be related with specifying trajectory setpoint
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
                    param2=self.PX4_CUSTOM_MAIN_MODE_AUTO_, \
                    param3=self.PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF_)
                self.get_logger().info("Current Mode: TakeOff (Auto-TakeOff)")

            # transition:
            # Debug: When the multirotor lands on the point which deviates from the original launch
            # point, the local position subscriber is unable to subscribe the local position topic
            # possibly, originates from the initialization problem of EKF2
            if (self.status_takeoff_ == self.TAKEOFF_STATE_FLIGHT_) \
                and (self.vehicle_local_position_z_ < -numpy.float32(2.0)):

                # exit:
                self.flight_phase_          =	2
                self.entry_execute_         =	1

        # Phase 2: Climb - Switch to offboard/position mode
        # Ascend to designated altitude of the takeoff completion point (TCP)
        # Exit when the multicoptor is located within the acceptance horizontal/vertical radius
        if self.flight_phase_ == 2:

            # entry:_mission.count
            if self.entry_execute_:

                self.entry_execute_ 			= 	0
                self.temp_callback_counter_		=   0
                self.offboard_ctrl_position_    =	True
                self.offboard_ctrl_attitude_    =	False
                self.trajectory_setpoint_x_     =   0.0
                self.trajectory_setpoint_y_     =   0.0
                self.trajectory_setpoint_z_     =   -20.0
                self.trajectory_setpoint_yaw_   =   0.0

            # during:
            self.get_logger().info("Current Mode: Climb (20m above from launch point)")
            self.publish_trajectory_setpoint()	
            self.publish_offboard_control_mode()

            if (self.status_nav_state_ != self.NAVIGATION_STATE_OFFBOARD_):
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
                    param2=self.PX4_CUSTOM_MAIN_MODE_OFFBOARD_)

            # transition:
            dist_xy =   numpy.sqrt(numpy.power(self.trajectory_setpoint_x_- \
                            self.vehicle_local_position_x_,2)+numpy.power( \
                            self.trajectory_setpoint_y_-self.vehicle_local_position_y_,2))
            dist_z  =   numpy.abs(self.trajectory_setpoint_z_-self.vehicle_local_position_z_)

            if (dist_xy < self.nav_xy_accept_rad_) and (dist_z < self.nav_z_accept_rad_) and \
                (numpy.absolute(self.vehicle_local_position_vz_) < self.nav_vz_accept_bound_):

                self.external_autopilot_engage_ =   1

                print(self.external_autopilot_ready_)
                print(self.external_controller_ready_)

                if (self.external_autopilot_ready_ == 1) and (self.external_controller_ready_ == 1):

                    # exit:
                    self.flight_phase_          =	3
                    self.entry_execute_         =	1

        # Phase 3: WPTs Mission - Switch to offboard/attitude mode
        # Perform waypoint mission based on external autopilot (acceleration/attitude command)
        # Exit when the multicoptor reaches a last waypoint and the autopilot turns off
        if self.flight_phase_ == 3:

            # entry:
            if self.entry_execute_:

                self.entry_execute_ 			= 	0
                self.temp_callback_counter_		=   0
                self.offboard_ctrl_position_    =	False
                self.offboard_ctrl_attitude_    =	True

            # during:
            self.get_logger().info("Current Mode: Offboard with waypoints")

            self.publish_offboard_control_mode()

            if (self.status_nav_state_ != self.NAVIGATION_STATE_OFFBOARD_):
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
                    param2=self.PX4_CUSTOM_MAIN_MODE_OFFBOARD_)

            # transition
            if (self.external_autopilot_ready_ == 0) or (self.external_controller_ready_ == 0):

                # exit:
                self.flight_phase_          =	4
                self.entry_execute_         =	1
                
        # Phase 4: Return - Switch to auto/return mode

        if self.flight_phase_ == 4:

            # entry:
            if self.entry_execute_:

                self.entry_execute_ 			= 	0
                self.temp_callback_counter_		=   0

            # during:
            self.get_logger().info("Current Mode: Return to Launch Site")

            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1, \
                    param2=self.PX4_CUSTOM_MAIN_MODE_AUTO_, \
                    param3=self.PX4_CUSTOM_SUB_MODE_AUTO_RTL_)

            # transition:

            if 0:

                # exit:
                self.flight_phase_          =	5
                self.entry_execute_         =	1

        self.temp_callback_counter_             =   self.temp_callback_counter_+1
        self.total_callback_counter_            =   self.total_callback_counter_+1

        print(self.wpts_targeted_offboard_[0,:])
        print(self.wpts_targeted_offboard_[1,:])
        print(self.wpts_targeted_offboard_[2,:])

    # Publisher functions
    def publish_vehicle_command(self,command,param1=float(0.0),param2=float(0.0),param3=float(0.0)):
        msg                                         =   VehicleCommand()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.param1                                  =   float(param1)
        msg.param2                                  =   float(param2)
        msg.param3                                  =   float(param3)
        msg.command                                 =   command
        msg.target_system                           =   1
        msg.target_component                        =   1
        msg.source_system                           =   1
        msg.source_component                        =   1
        msg.from_external                           =   True

        self.vehicle_command_publisher_.publish(msg)

    def publish_trajectory_setpoint(self):
        msg                                         =   TrajectorySetpoint()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.position[0]                             =   float(self.trajectory_setpoint_x_)
        msg.position[1]                             =   float(self.trajectory_setpoint_y_)
        msg.position[2]                             =   float(self.trajectory_setpoint_z_)
        msg.yaw                                     =   float(self.trajectory_setpoint_yaw_)

        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_offboard_control_mode(self):
        msg                                         =   OffboardControlMode()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.position                                =   self.offboard_ctrl_position_ 
        msg.velocity                                =   self.offboard_ctrl_velocity_ 
        msg.acceleration                            =   self.offboard_ctrl_acceleration_
        msg.attitude                                =   self.offboard_ctrl_attitude_
        msg.body_rate                               =   self.offboard_ctrl_body_rate_

        self.offboard_control_mode_publisher_.publish(msg)

    def publish_fm_status(self):
        msg                                         =   ExtGCUFlightManagementStatus()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        # msg.flight_phase                            =   int(self.flight_phase_)
        msg.external_autopilot_engage               =   int(self.external_autopilot_engage_)

        self.fm_status_publisher_.publish(msg)

    # Subscriber functions
    def process_timesync(self,msg):
        self.timesync_timestamp_                    =   numpy.uint64(msg.timestamp)

    def process_vehicle_status(self,msg):
        self.status_nav_state_                      =   numpy.uint8(msg.nav_state)
        self.status_arming_status_                  =   numpy.uint8(msg.arming_state)

    def process_takeoff_status(self,msg):
        self.status_takeoff_                        =   numpy.uint8(msg.takeoff_state)

    def process_vehicle_local_position(self,msg):
        self.vehicle_local_position_x_              =   numpy.float32(msg.x)
        self.vehicle_local_position_y_              =   numpy.float32(msg.y)
        self.vehicle_local_position_z_              =   numpy.float32(msg.z)
        self.vehicle_local_position_vx_             =   numpy.float32(msg.vx)
        self.vehicle_local_position_vy_             =   numpy.float32(msg.vy)
        self.vehicle_local_position_vz_             =   numpy.float32(msg.vz)

    def process_autopilot_status(self,msg):
        self.external_autopilot_ready_              =   numpy.uint8(msg.external_autopilot_ready)

    def process_controller_status(self,msg):
        self.external_controller_ready_             =   numpy.uint8(msg.external_controller_ready)

    def process_vehicle_trajectory(self,msg):
        for idx in range(3):
            self.wpts_targeted_offboard_[idx,:]     =   msg.waypoints[idx].position

        self.last_wpt_call_                         =   msg.last_wpt_call
        self.wpt_update_reply_                      =   msg.wpt_update_reply
        self.external_wpt_manager_ready_            =   msg.external_wpt_manager_ready

def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create extGCU attitude control node
    extgcu_fm_module_           =	extgcu_fm_module()

    # Spin the created control node
    rclpy.spin(extgcu_fm_module_)

    # After spinning, destroy the node and shutdown rclpy library
    extgcu_fm_module_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()