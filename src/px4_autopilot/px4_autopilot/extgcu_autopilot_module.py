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
from extgcu_library import autopilot
from extgcu_library import get_3D_to_AzEl, get_AzEl_to_3D, get_Transform_3D, Integrate_Euler

# Define flight management module using Python class
class extgcu_autopilot_module(Node,autopilot):

    def __init__(self):

        # Inherit from the parent class 'Node'
        super().__init__('extgcu_autopilot_module')

        # Load parameters from the parent class
        super().load_parameters()

        # Define publishers
        self.acceleration_command_publisher_        =   self.create_publisher(
            ExtGCUAccelerationCommand,
            'extgcu/acceleration',
            10
        )
        self.autopilot_status_publisher_            =   self.create_publisher(
            ExtGCUAutopilotStatus,
            'extgcu/autopilot_status',
            10
        )

        # Define subscribers
        self.timesync_subscriber_                   =   self.create_subscription(
            Timesync,
            'fmu/timesync/out',
            self.process_timesync,
            10
        )
        self.vehicle_local_position_subscriber_     =   self.create_subscription(
            VehicleLocalPosition,
            'fmu/vehicle_local_position/out',
            self.process_vehicle_local_position,
            10
        )
        self.vehicle_trajectory_subscriber_         =   self.create_subscription(
            ExtGCUTrajectory,
            'extgcu/trajectory',
            self.process_vehicle_trajectory,
            10
        )
        self.fm_status_subscriber_                  =   self.create_subscription(
            ExtGCUFlightManagementStatus,
            'extgcu/fm_status',
            self.process_fm_status,
            10
        )

        # Initialize parameters for timer callback function
        self.timer_period_                      =   numpy.float32(self.ap_dt_)
        self.timer_                             =   self.create_timer(self.timer_period_, \
                                                        self.autopilot_callback)
        self.temp_callback_counter_             =   numpy.uint64(0)
        self.total_callback_counter_            =   numpy.uint64(0)
        self.timestamp_checker_                 =	numpy.uint64(0)

        # Initialize parameters for autopilot module
        self.entry_execute_                     =	numpy.uint8(1)
        self.persistent_counter_ 	            =	numpy.uint8(0)
        self.persistent_threshold_              =   numpy.uint8(self.persistent_timer_ \
                                                        /self.timer_period_)

        # Initialize fields of acceleration command topic
        self.acceleration_command_              =   numpy.zeros(3,dtype=numpy.float32)
 
        # Initialize fields of autopilot status topic
        self.wpt_update_request_                =   numpy.uint8(0)
        self.external_autopilot_ready_          =   numpy.uint8(0)
        
        # Initialize fields of timesync topic
        self.timesync_timestamp_                =   numpy.uint64(0)

        # Initialize fields of local position topic
        self.vehicle_local_position_x_          =   numpy.float32(0.0)
        self.vehicle_local_position_y_          =   numpy.float32(0.0)
        self.vehicle_local_position_z_          =   numpy.float32(0.0)
        self.vehicle_local_position_vx_         =   numpy.float32(0.0)
        self.vehicle_local_position_vy_         =   numpy.float32(0.0)
        self.vehicle_local_position_vz_         =   numpy.float32(0.0)
        self.vehicle_local_position_yaw_        =   numpy.float32(0.0)
        
        # Initialize fields of vehicle trajectory topic
        self.last_wpt_call_                     =   numpy.uint8(0)
        self.wpt_update_reply_                  =   numpy.uint8(0)
        self.external_wpt_manager_ready_        =   numpy.uint8(0)
        self.targeted_wpts_position_            =   numpy.zeros((3,3),dtype=numpy.float32)
        self.targeted_wpts_position_memory_     =   numpy.zeros((3,3),dtype=numpy.float32)

        # Initialize fields of flight management status topic
        self.external_autopilot_engage_         =   numpy.uint8(0)

        # Declare subscribers once to avoid unused variable warning
        self.timesync_subscriber_
        self.vehicle_local_position_subscriber_
        self.vehicle_trajectory_subscriber_
        self.fm_status_subscriber_

    # Autopilot module callback function
    def autopilot_callback(self):

        # Create a log message at NuttShell
        self.get_logger().info('Autopilot Callback')

        # Persistent counter: increase counter number when on, reset counter to zero when off
        # Quadrotor switches to offboard/attitude mode when persistent counter exceeds threshold
        if (self.external_autopilot_engage_ == 1) and (self.last_wpt_call_ != 1):

            if self.persistent_counter_ < self.persistent_threshold_:
                self.persistent_counter_            +=  1

            else:
                self.external_autopilot_ready_      =   1
        
        elif (self.external_autopilot_engage_ == 1) and (self.last_wpt_call_ == 1):

            rel_dist_wpt          =   numpy.sqrt(numpy.power(self.targeted_wpts_position_[1,0]-self.X_virtual_[0],2)+ \
                                            numpy.power(self.targeted_wpts_position_[1,1]-self.X_virtual_[1],2)+ \
                                            numpy.power(self.targeted_wpts_position_[1,2]-self.X_virtual_[2],2))

            if rel_dist_wpt < numpy.float32(15):
                self.external_autopilot_ready_      =   0

            self.persistent_counter_    =   0

        # Autopilot (guidance law)
        if (self.external_autopilot_engage_ == 1) and (self.external_wpt_manager_ready_ == 1):

            # Initialize autopilot
            if self.entry_execute_ == 1:

                self.entry_execute_     =   0
                self.initialize_virtual_target()

            # Update kinematic variables and get acceleration command using guidance law
            self.virtual_target_update()
            self.kinematics_update()

            # Obtain axial and orthogonal acceleration command (PPN form)
            tangential_acc  =   numpy.reshape(self.get_velocity_hold_cmd(),(1,))
            orthogonal_acc  =   self.get_guidance_cmd()

            total_acc       =   numpy.concatenate((tangential_acc,orthogonal_acc))

            vel_vehicle_3d          =   [self.vehicle_local_position_vx_, \
                                        self.vehicle_local_position_vy_, \
                                        self.vehicle_local_position_vz_]

            psigam_vehicle_3d       =   get_3D_to_AzEl(vel_vehicle_3d)

            self.acceleration_command_  =   numpy.matmul(numpy.linalg.inv( \
                                                get_Transform_3D(psigam_vehicle_3d)), \
                                                total_acc)

            # Publish check
            print('==============================')
            print('Acceleration in wind frame [m]: %.2f, %.2f, %.2f' \
                    %(tangential_acc,orthogonal_acc[0],orthogonal_acc[1]))
            print('Acceleration in NED frame [m]: %.2f, %.2f, %.2f' \
                    %(self.acceleration_command_[0],self.acceleration_command_[1], \
                      self.acceleration_command_[2]))

            self.publish_acceleration_command()
            self.publish_autopilot_status()

            self.total_callback_counter_    =   self.total_callback_counter_+1

    # Virtual target initialization
    def initialize_virtual_target(self):

        # Vehicle speed (set lowest limit to be 0.1m/s)
        spd_vehicle_3d          =   numpy.sqrt(numpy.power(self.vehicle_local_position_vx_,2)+ \
                                        numpy.power(self.vehicle_local_position_vy_,2)+ \
                                        numpy.power(self.vehicle_local_position_vz_,2))
        spd_vehicle_3d          =   numpy.max(numpy.array([spd_vehicle_3d,self.V_lbound_]))

        # Initialize look ahead distance
        self.LA_dist_           =   self.T_horizon_*spd_vehicle_3d

        # Get initial position of virtual target from initial waypoints information
        psigam_ref              =   get_3D_to_AzEl(self.targeted_wpts_position_[1,:] \
                                                        -self.targeted_wpts_position_[0,:])   # index -1

        self.X_virtual_[0:3]    =   self.targeted_wpts_position_[0,:]+ \
                                        spd_vehicle_3d*get_AzEl_to_3D(psigam_ref)*self.T_horizon_ # index -1

        # Calculate relative position between virtual target and vehicle
        self.R_rel_[0]          =   self.X_virtual_[0]-self.vehicle_local_position_x_
        self.R_rel_[1]          =   self.X_virtual_[1]-self.vehicle_local_position_y_
        self.R_rel_[2]          =   self.X_virtual_[2]-self.vehicle_local_position_z_

        self.R_rel_norm_        =   numpy.linalg.norm(self.R_rel_)

        # Get initial velocity of virtual target from vehicle speed
        self.X_virtual_[3:6]    =   spd_vehicle_3d*self.LA_dist_/max(self.R_rel_norm_,self.eps_)* \
                                        get_AzEl_to_3D(psigam_ref)

        vel_target_3d           =   spd_vehicle_3d*get_AzEl_to_3D(psigam_ref)
        vel_vehicle_3d          =   spd_vehicle_3d*get_AzEl_to_3D(get_3D_to_AzEl( \
                                        [self.vehicle_local_position_vx_, \
                                         self.vehicle_local_position_vy_, \
                                         self.vehicle_local_position_vz_]))

        # Calculate relative velocity between virtual target and vehicle
        # Note that the vehicle velocity is set to be non-zero value
        self.V_rel_[0]          =   vel_target_3d[0]-vel_vehicle_3d[0]
        self.V_rel_[1]          =   vel_target_3d[1]-vel_vehicle_3d[1]
        self.V_rel_[2]          =   vel_target_3d[2]-vel_vehicle_3d[2]

        self.V_UAV_norm_        =   spd_vehicle_3d

        # Calculate close speed
        self.Vclose_            =   -numpy.dot(self.R_rel_,self.V_rel_)/self.R_rel_norm_

        # Estimate time-to-go
        self.Tgo_               =   self.R_rel_norm_/self.V_UAV_norm_

        # Estimate line-of-sight in 3D Cartesian coordinate
        self.Lambda             =   get_3D_to_AzEl(self.R_rel_)

        # Save targeted waypoints information
        self.targeted_wpts_position_memory_     =   self.targeted_wpts_position_

    # Update kinematics of vehicle and virtual target
    def kinematics_update(self):

        # Vehicle speed (set lowest limit to be 0.1m/s)
        spd_vehicle_3d          =   numpy.sqrt(numpy.power(self.vehicle_local_position_vx_,2)+ \
                                        numpy.power(self.vehicle_local_position_vy_,2)+ \
                                        numpy.power(self.vehicle_local_position_vz_,2))
        spd_vehicle_3d          =   numpy.max(numpy.array([spd_vehicle_3d,self.V_lbound_]))

        vel_vehicle_3d          =   spd_vehicle_3d*get_AzEl_to_3D(get_3D_to_AzEl( \
                                        [self.vehicle_local_position_vx_, \
                                         self.vehicle_local_position_vy_, \
                                         self.vehicle_local_position_vz_]))

        # Update look ahead distance
        self.LA_dist_           =   self.T_horizon_*spd_vehicle_3d   

        # Update relative states between virtual target and vehicle
        self.R_rel_[0]          =   self.X_virtual_[0]-self.vehicle_local_position_x_
        self.R_rel_[1]          =   self.X_virtual_[1]-self.vehicle_local_position_y_
        self.R_rel_[2]          =   self.X_virtual_[2]-self.vehicle_local_position_z_

        self.V_rel_[0]          =   self.X_virtual_[3]-vel_vehicle_3d[0]
        self.V_rel_[1]          =   self.X_virtual_[4]-vel_vehicle_3d[1]
        self.V_rel_[2]          =   self.X_virtual_[5]-vel_vehicle_3d[2]

        self.R_rel_norm_        =   max(numpy.linalg.norm(self.R_rel_),self.eps_)

        self.V_UAV_norm_        =   spd_vehicle_3d

        # Calculate close speed
        self.Vclose_            =   -numpy.dot(self.R_rel_,self.V_rel_)/self.R_rel_norm_

        # Estimate time-to-go
        self.Tgo_               =   self.R_rel_norm_/self.V_UAV_norm_

        # Estimate line-of-sight in 3D Cartesian coordinate
        self.Lambda             =   get_3D_to_AzEl(self.R_rel_)

    # Update virtual target states
    def virtual_target_update(self):

        pos_vehicle_3d          =   [self.vehicle_local_position_x_, \
                                     self.vehicle_local_position_y_, \
                                     self.vehicle_local_position_z_]

        spd_vehicle_3d          =   numpy.sqrt(numpy.power(self.vehicle_local_position_vx_,2)+ \
                                        numpy.power(self.vehicle_local_position_vy_,2)+ \
                                        numpy.power(self.vehicle_local_position_vz_,2))
        spd_vehicle_3d          =   numpy.max(numpy.array([spd_vehicle_3d,self.V_lbound_]))

        if self.publish_mode_ == 0:

            if numpy.linalg.norm(self.targeted_wpts_position_[1,:]-pos_vehicle_3d) < self.LA_dist_: # index -1

                self.wpt_update_request_    =   1

            if (self.wpt_update_request_ == 1) and (self.wpt_update_reply_ == 1):

                self.wpt_update_request_    =   0

                psigam_ref              =   get_3D_to_AzEl(self.targeted_wpts_position_[1,:]- \
                                                self.targeted_wpts_position_[0,:]) # index -1
                self.X_virtual_[0:3]    =   self.targeted_wpts_position_[0,:]+ \
                                                self.X_virtual_[3:6]*self.ap_dt_ # index -1
                self.X_virtual_[3:6]    =   spd_vehicle_3d*self.LA_dist_/max(self.R_rel_norm_, \
                                                self.eps_)*get_AzEl_to_3D(psigam_ref)

            else:

                self.X_virtual_[0:3]    =   self.X_virtual_[0:3]+self.X_virtual_[3:6]*self.ap_dt_
                self.X_virtual_[3:6]    =   spd_vehicle_3d*self.LA_dist_/max(self.R_rel_norm_, \
                                                self.eps_)*get_AzEl_to_3D( \
                                                get_3D_to_AzEl(self.X_virtual_[3:6]))

        elif self.publish_mode_ == 2:

            if numpy.linalg.norm(self.targeted_wpts_position_[1,:]-self.targeted_wpts_position_memory_[1,:]) > self.eps_*self.V_desired_: # index -1

                psigam_ref              =   get_3D_to_AzEl(self.targeted_wpts_position_[1,:]- \
                                                self.targeted_wpts_position_[0,:]) # index -1
                self.X_virtual_[0:3]    =   self.targeted_wpts_position_[0,:]+ \
                                                self.X_virtual_[3:6]*self.ap_dt_ # index -1
                self.X_virtual_[3:6]    =   spd_vehicle_3d*self.LA_dist_/max(self.R_rel_norm_, \
                                                self.eps_)*get_AzEl_to_3D(psigam_ref)

            else:

                self.X_virtual_[0:3]    =   self.X_virtual_[0:3]+self.X_virtual_[3:6]*self.ap_dt_
                self.X_virtual_[3:6]    =   spd_vehicle_3d*self.LA_dist_/max(self.R_rel_norm_, \
                                                self.eps_)*get_AzEl_to_3D( \
                                                get_3D_to_AzEl(self.X_virtual_[3:6]))

    def get_guidance_cmd(self):

        vel_vehicle_3d          =   [self.vehicle_local_position_vx_, \
                                     self.vehicle_local_position_vy_, \
                                     self.vehicle_local_position_vz_]

        psigam_vehicle_3d       =   get_3D_to_AzEl(vel_vehicle_3d)

        self.lead_angle_        =   self.Lambda-psigam_vehicle_3d

        self.lead_angle_[0]     =   math.atan2(math.sin(self.lead_angle_[0]), \
                                        math.cos(self.lead_angle_[0]))
        self.lead_angle_[1]     =   math.atan2(math.sin(self.lead_angle_[1]), \
                                        math.cos(self.lead_angle_[1]))

        ortho_acc_cmd           =   numpy.empty(2,dtype=numpy.float32)

        ortho_acc_cmd[0]        =   self.eta_*self.V_UAV_norm_*math.cos(psigam_vehicle_3d[1]) \
                                        /self.Tgo_*self.lead_angle_[0]
        ortho_acc_cmd[1]        =   -self.eta_*self.V_UAV_norm_/self.Tgo_*self.lead_angle_[1]

        const_normalize         =   numpy.linalg.norm(ortho_acc_cmd)/(1.5*self.gravity_const_)

        ortho_acc_cmd           =   ortho_acc_cmd/max(1,const_normalize)

        return ortho_acc_cmd

    def get_velocity_hold_cmd(self):

        spd_vehicle_3d          =   numpy.sqrt(numpy.power(self.vehicle_local_position_vx_,2)+ \
                                        numpy.power(self.vehicle_local_position_vy_,2)+ \
                                        numpy.power(self.vehicle_local_position_vz_,2))

        self.vel_err_           =   self.V_desired_-spd_vehicle_3d
        self.int_vel_err_       =   Integrate_Euler(self.vel_err_,self.int_vel_err_,self.ap_dt_)

        axial_acc_cmd           =   self.vel_err_*self.Kp_vel_+self.int_vel_err_*self.Ki_vel_

        return axial_acc_cmd

    # Publisher functions
    def publish_acceleration_command(self):
        msg                                     =   ExtGCUAccelerationCommand()
        msg.timestamp                           =   int(self.timesync_timestamp_)
        msg.acceleration                        =   numpy.array(self.acceleration_command_,dtype=numpy.float32) # numpy.array([0.0,1.0,-1.0],dtype=numpy.float32) 

        self.acceleration_command_publisher_.publish(msg)

    def publish_autopilot_status(self):
        msg                                     =   ExtGCUAutopilotStatus()
        msg.timestamp                           =   int(self.timesync_timestamp_)
        msg.wpt_update_request                  =   int(self.wpt_update_request_)
        msg.external_autopilot_ready            =   int(self.external_autopilot_ready_)

        self.autopilot_status_publisher_.publish(msg)

    # Subscriber functions
    def process_timesync(self,msg):
        self.timesync_timestamp_                    =   numpy.uint64(msg.timestamp)

    def process_vehicle_local_position(self,msg):
        self.vehicle_local_position_x_              =   numpy.float32(msg.x)
        self.vehicle_local_position_y_              =   numpy.float32(msg.y)
        self.vehicle_local_position_z_              =   numpy.float32(msg.z)
        self.vehicle_local_position_vx_             =   numpy.float32(msg.vx)
        self.vehicle_local_position_vy_             =   numpy.float32(msg.vy)
        self.vehicle_local_position_vz_             =   numpy.float32(msg.vz)
        self.vehicle_local_position_yaw_            =   numpy.float32(msg.heading)

    def process_vehicle_trajectory(self,msg):

        for idx in range(3):
            self.targeted_wpts_position_[idx,:]     =   msg.waypoints[idx].position

        self.last_wpt_call_                         =   msg.last_wpt_call
        self.wpt_update_reply_                      =   msg.wpt_update_reply
        self.external_wpt_manager_ready_            =   msg.external_wpt_manager_ready
        self.publish_mode_                          =   msg.publish_mode

    def process_fm_status(self,msg):
        self.external_autopilot_engage_             =   numpy.uint8(msg.external_autopilot_engage)

def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create extGCU attitude control node
    extgcu_autopilot_module_    =   extgcu_autopilot_module()

    # Spin the created control node
    rclpy.spin(extgcu_autopilot_module_)

    # After spinning, destroy the node and shutdown rclpy library
    extgcu_autopilot_module_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()