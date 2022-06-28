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
from extgcu_library import wpt_manager

# Define flight management module using Python class
class extgcu_wpt_manage_module(Node,wpt_manager):

    def __init__(self):

        # Inherit from the parent class 'Node'
        super().__init__('extgcu_wpt_manage_module')

        # Load parameters from the parent class
        super().load_parameters()

        # Define publishers
        self.trajectory_publisher_toAP_             =   self.create_publisher(
            ExtGCUTrajectory,
            'extgcu/trajectory',
            10
        )
        self.trajectory_publisher_toPX4_            =   self.create_publisher(
            VehicleTrajectoryWaypoint,
            "fmu/vehicle_trajectory_waypoint/in",
            10
        )
        self.telemetry_status_publisher_            =   self.create_publisher(
            TelemetryStatus,
            'fmu/telemetry_status/in',
            10
        )
        self.offboard_custom_engage_publisher_      =   self.create_publisher(
            OffboardEngage,
            'fmu/offboard_engage/in',
            10
        )

        # Define subscribers
        self.timesync_subscriber_                   =   self.create_subscription(
            Timesync,
            'fmu/timesync/out',
            self.process_timesync,
            10
        )
        self.fm_status_subscriber_                  =   self.create_subscription(
            ExtGCUFlightManagementStatus,
            'extgcu/fm_status',
            self.process_fm_status,
            10
        )
        self.trajecotry_desired_subscriber_fromPX4_ =   self.create_subscription(
            VehicleTrajectoryWaypointDesired,
            'fmu/vehicle_trajectory_waypoint_desired/out',
            self.process_desired_trajectory_waypoint_fromPX4, 
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
        self.offboard_trajectory_subscriber_        =   self.create_subscription(
            OffboardTrajectory,
            'fmu/offboard_trajectory/out',
            self.process_offboard_trajectory,
            10
        )

        # Initialize parameters for timer callback function
        self.timer_period_                      =   numpy.float32(self.wpt_dt_)
        self.timer                              =   self.create_timer(self.timer_period_, \
                                                        self.wpt_callback)
        self.temp_callback_counter_             =   numpy.uint64(0)
        self.total_callback_counter_            =   numpy.uint64(0)
        self.timestamp_checker_                 =	numpy.uint64(0)

        # Initialize parameters for waypoint management module
        self.entry_execute_                     =	numpy.uint8(1)
        self.persistent_counter_ 	            =	numpy.uint8(0)
        self.persistent_threshold_              =   numpy.uint8(self.persistent_timer_ \
                                                        /self.timer_period_)
        self.wpt_source_                        =   numpy.uint8(0)

        # Waypoint source
        # 0: Load waypoints information from code/file
        if self.wpt_source_ == 0:

            self.position_capture_                  =   numpy.empty(3,dtype=numpy.float32)
            self.wpt_cur_ID_                        =   numpy.uint8(0)
            self.num_wpts_saved_                    =   6

            self.wpts_saved_                        =   numpy.empty((self.num_wpts_saved_,3), \
                                                                    dtype=numpy.float32)
            self.wpts_saved_[0,0:3]                 =   numpy.array([0.0,0.0,-50.0])
            self.wpts_saved_[0,0:3]                 =   numpy.array([0.0,20.0,-50.0])
            self.wpts_saved_[1,0:3]                 =   numpy.array([100.0,20.0,-60.0])
            self.wpts_saved_[2,0:3]                 =   numpy.array([100.0,120.0,-70.0])
            self.wpts_saved_[3,0:3]                 =   numpy.array([0.0,120.0,-60.0])
            self.wpts_saved_[4,0:3]                 =   numpy.array([0.0,20.0,-50.0])

        # 1: Read waypoints information from external offboard guidance module (Aware4 project)
        #    Saved waypoints will be transferred from path planning module
        #    (TBW)
        elif self.wpt_source_ == 1:

            print(" ")

        # 2: Read waypoints information from PX4 uORB message (Custom OffboardTrajectory msg)
        elif self.wpt_source_ == 2:

            self.wpts_desired_offboard_             =   numpy.zeros((3,3),dtype=numpy.float32)

        # Initialize fields of trajectory (to AP) topic
        self.last_wpt_call_                     =   numpy.uint8(0)
        self.wpt_update_reply_                  =   numpy.uint8(0)
        self.external_wpt_manager_ready_        =   numpy.uint8(0)
        self.targeted_wpts_position_            =   numpy.empty((3,3),dtype=numpy.float32)

        # Initialize fields of trajectory (to PX4) topic
        # Same as vehicle_trajectory_waypoint_desired topic

        # Initialize fields of telemetry topic
        self.telemtry_heartbeat_obs_avoidance_  =   True
        self.telemtry_avoidance_system_healty_  =   True

        # Initialize fields of offboard custom engage topic
        self.offboard_custom_engage_            =   numpy.uint8(0)

        # Initialize fields of timesync topic
        self.timesync_timestamp_                =   numpy.uint64(0)

        # Initialize fields of flight management status topic
        self.external_autopilot_engage_         =   numpy.uint8(0)

        # Initialize fields of trajectory waypoint desired topic (from PX4 or offboard guidance)
        self.desired_NUMBER_POINTS_             =   numpy.uint8(0)

        self.desired_wpts_position_             =   numpy.empty((5,3),dtype=numpy.float32)
        self.desired_wpts_velocity_             =   numpy.empty((5,3),dtype=numpy.float32)
        self.desired_wpts_acceleration_         =   numpy.empty((5,3),dtype=numpy.float32)
        self.desired_wpts_yaw_                  =   numpy.empty((5,1),dtype=numpy.float32)
        self.desired_wpts_yaw_speed_            =   numpy.empty((5,1),dtype=numpy.float32)
        self.desired_wpts_validity_             =   numpy.empty((5,1),dtype=numpy.bool8)
        self.desired_wpts_type_                 =   numpy.empty((5,1),dtype=numpy.uint8)

        # Initialize fields of local position topic
        self.vehicle_local_position_x_          =   numpy.float32(0.0)
        self.vehicle_local_position_y_          =   numpy.float32(0.0)
        self.vehicle_local_position_z_          =   numpy.float32(0.0)
        self.vehicle_local_position_yaw_        =   numpy.float32(0.0)

        # Initialize fields of autopilot status topic
        self.wpt_update_request_                =   numpy.uint8(0)

        # Initialize fields of custom offboard trajectory desired topic
        self.offboard_custom_ready_             =   numpy.uint8(0)
        self.offboard_last_wpt_call_            =   numpy.uint8(0)

        # Declare subscribers once to avoid unused variable warning
        self.timesync_subscriber_
        self.fm_status_subscriber_
        self.trajecotry_desired_subscriber_fromPX4_
        self.vehicle_local_position_subscriber_
        self.autopilot_status_subscriber_
        self.offboard_trajectory_subscriber_

    # Waypoint management module callback function
    def wpt_callback(self):
        
        # Create a log message at NuttShell
        self.get_logger().info('Waypoint Manager Callback')

        # If the quadrotor is operated using full offboard, the trajectory (waypoints) information
        # will be provided by guidance module. (Guidance + Obstacle Detection + Collision avoidance)
        # Process received waypoints information for ExtGCU Autopilot module

        # For internal module function tests, load saved waypoints in this code
        if self.wpt_source_ == 0:

            if self.external_autopilot_engage_ == 1:

                if self.persistent_counter_ < self.persistent_threshold_:
                    self.persistent_counter_            +=  1

                else:
                    self.external_wpt_manager_ready_    =   1
            
            else:
                self.persistent_counter_            =   0
                self.external_wpt_manager_ready_    =   0
                self.wpt_cur_ID_                    =   0
                self.entry_execute_                 =   1
            
            if self.external_autopilot_engage_ == 1:

                if self.entry_execute_ == 1:
                    self.entry_execute_         =   0
                    self.position_capture_[0]   =   self.vehicle_local_position_x_
                    self.position_capture_[1]   =   self.vehicle_local_position_y_
                    self.position_capture_[2]   =   self.vehicle_local_position_z_

                # Check that current waypoint ID is less or equal to total waypoint number
                assert self.wpt_cur_ID_ <= self.num_wpts_saved_, \
                            'Check Waypoint ID of Waypoint Manager'

                if (self.wpt_update_request_ == 1) and (self.wpt_update_reply_ == 0) \
                    and (self.last_wpt_call_ != 1):
                    
                    if self.wpt_cur_ID_ < self.num_wpts_saved_-2:
                        self.wpt_cur_ID_                =   self.wpt_cur_ID_+1

                    else:
                        self.wpt_cur_ID_                =   self.wpt_cur_ID_+1
                        self.last_wpt_call_             =   1

                    self.wpt_update_reply_      =   1

                elif (self.wpt_update_request_ == 0) and (self.wpt_update_reply_ == 1):
                    self.wpt_update_reply_      =   0

                if self.wpt_cur_ID_ == 0:
                    self.targeted_wpts_position_[0,0:3] =   self.position_capture_
                    self.targeted_wpts_position_[1,0:3] =   self.wpts_saved_[self.wpt_cur_ID_,0:3]
                    self.targeted_wpts_position_[2,0:3] =   numpy.ones((1,3), \
                                                                dtype=numpy.float32)*numpy.nan

                else:
                    self.targeted_wpts_position_[0:2,0:3]    \
                        =   self.wpts_saved_[self.wpt_cur_ID_-1:self.wpt_cur_ID_+1,0:3]
                    self.targeted_wpts_position_[2,0:3] =   numpy.ones((1,3),dtype=numpy.float32)* \
                                                                numpy.nan

            self.publish_trajecotry_wpts_toAP()

        # For PX4 simulation environment integrated tests, get PX4 internal waypoints information
        elif self.wpt_source_ == 2:

            if self.external_autopilot_engage_ == 1:
                self.offboard_custom_engage_    =   1
            
            else:
                self.offboard_custom_engage_    =   0

            if self.offboard_custom_ready_ == 1:

                if self.persistent_counter_ < self.persistent_threshold_:
                    self.persistent_counter_            +=  1

                else:
                    self.external_wpt_manager_ready_    =   1

            else:
                self.persistent_counter_            =   0
                self.external_wpt_manager_ready_    =   0
                self.entry_execute_                 =   1

            if self.external_autopilot_engage_ == 1:
                self.targeted_wpts_position_        =   self.wpts_desired_offboard_

            else:
                self.targeted_wpts_position_        =   numpy.ones((3,3), \
                                                                dtype=numpy.float32)*numpy.nan

            self.publish_trajecotry_wpts_toAP()
            self.publish_offboard_custom_engage()
        
            # self.publish_trajecotry_wpts_toPX4()
            # self.publish_telem`try_status_toPX4()

    # Publisher functions
    def publish_trajecotry_wpts_toAP(self):
        msg                                         =   ExtGCUTrajectory()

        for idx in range(3):
            msg.waypoints[idx].position             =   self.targeted_wpts_position_[idx,:]

        msg.last_wpt_call                           =   int(self.last_wpt_call_)
        msg.wpt_update_reply                        =   int(self.wpt_update_reply_)
        msg.external_wpt_manager_ready              =   int(self.external_wpt_manager_ready_)
        msg.publish_mode                            =   int(self.wpt_source_)

        self.trajectory_publisher_toAP_.publish(msg)

    def publish_trajecotry_wpts_toPX4(self):
        msg                                         =   VehicleTrajectoryWaypoint()

        for idx in range(1):
            msg.waypoints[idx].position             =   self.desired_wpts_position_[idx,:]
            msg.waypoints[idx].velocity             =   self.desired_wpts_velocity_[idx,:]
            msg.waypoints[idx].acceleration         =   self.desired_wpts_acceleration_[idx,:]
            msg.waypoints[idx].yaw                  =   float(self.desired_wpts_yaw_[idx])
            msg.waypoints[idx].yaw_speed            =   float(self.desired_wpts_yaw_speed_[idx])
            msg.waypoints[idx].type                 =   int(self.desired_wpts_type_[idx])
            msg.waypoints[idx].point_valid          =   bool(self.desired_wpts_validity_[idx])

        self.trajectory_publisher_toPX4_.publish(msg)

    def publish_telemtry_status_toPX4(self):
        msg                                         =   TelemetryStatus()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.heartbeat_component_obstacle_avoidance  =   self.telemtry_heartbeat_obs_avoidance_
        msg.avoidance_system_healthy                =   self.telemtry_avoidance_system_healty_

        self.telemetry_status_publisher_.publish(msg)

    def publish_offboard_custom_engage(self):
        msg                                         =   OffboardEngage()
        msg.timestamp                               =   int(self.timesync_timestamp_)
        msg.offboard_custom_engage                  =   int(self.offboard_custom_engage_)
        print(msg.offboard_custom_engage)
        
        self.offboard_custom_engage_publisher_.publish(msg)

    # Subscriber functions
    def process_timesync(self,msg):
        self.timesync_timestamp_                    =   numpy.uint64(msg.timestamp)

    def process_fm_status(self,msg):
        self.external_autopilot_engage_             =   numpy.uint8(msg.external_autopilot_engage)

    def process_desired_trajectory_waypoint_fromPX4(self,msg):
        self.desired_NUMBER_POINTS_                 =   numpy.uint8(msg.NUMBER_POINTS)

        for idx in range(msg.NUMBER_POINTS):
            self.desired_wpts_position_[idx,:]      =   numpy.float32(msg.waypoints[idx].position)
            self.desired_wpts_velocity_[idx,:]      =   numpy.float32(msg.waypoints[idx].velocity)
            self.desired_wpts_acceleration_[idx,:]  =   numpy.float32(msg.waypoints[idx].acceleration)
            self.desired_wpts_yaw_[idx]             =   numpy.float32(msg.waypoints[idx].yaw)
            self.desired_wpts_yaw_speed_[idx]       =   numpy.float32(msg.waypoints[idx].yaw_speed)
            self.desired_wpts_type_[idx]            =   numpy.uint8(msg.waypoints[idx].type)
            self.desired_wpts_validity_[idx]        =   numpy.bool8(msg.waypoints[idx].point_valid)

    def process_vehicle_local_position(self,msg):
        self.vehicle_local_position_x_              =   numpy.float32(msg.x)
        self.vehicle_local_position_y_              =   numpy.float32(msg.y)
        self.vehicle_local_position_z_              =   numpy.float32(msg.z)

    def process_autopilot_status(self,msg):
        self.wpt_update_request_                    =   numpy.uint8(msg.wpt_update_request)

    def process_offboard_trajectory(self,msg):
        self.offboard_custom_ready_                 =   numpy.uint8(msg.offboard_custom_ready)

        self.wpts_desired_offboard_[0,0:3]          =   numpy.float32(msg.waypoints[0].position)
        self.wpts_desired_offboard_[1,0:3]          =   numpy.float32(msg.waypoints[1].position)
        self.wpts_desired_offboard_[2,0:3]          =   numpy.float32(msg.waypoints[2].position)

def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create extGCU attitude control node
    extgcu_wpt_manage_module_   =   extgcu_wpt_manage_module()

    # Spin the created control node
    rclpy.spin(extgcu_wpt_manage_module_)

    # After spinning, destroy the node and shutdown rclpy library
    extgcu_wpt_manage_module_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()