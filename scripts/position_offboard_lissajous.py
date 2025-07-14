#!/usr/bin/env python3

__author__ = "Minhyun Cho"
__contact__ = "@purdue.edu"

# python packages and modules
import argparse
import numpy as np
from functools import partial

# ros packages
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# messages
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleStatus, VehicleLocalPosition, VehicleGlobalPosition, VehicleCommand
from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray

class OffboardPositionLissajous(Node):

    def __init__(self,amplitude,frequency,phase,duration):
        # inputs
        # amplitude: np.ndarray of shape (4,), specifies amplitude of sine wave along each of 3 axes and yaw
        # frequency: np.ndarray of shape (4, 2), defines frequency sweep bounds [fmin, fmax] for each axis and yaw over time
        # phase: np.ndarray of shape (4,), specifies phase offset (in radians) for sine function on each axis and yaw
        # duration: np.float, total duration over which excitation signal is generated

        # inheritance from parent class
        super().__init__("position_offboard_sitl")

        # set publisher and subscriber quality of service profile
        qos_profile_pub     =   QoSProfile(
            reliability     =   QoSReliabilityPolicy.BEST_EFFORT,
            durability      =   QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history         =   QoSHistoryPolicy.KEEP_LAST,
            depth           =   1
        )

        qos_profile_sub     =   QoSProfile(
            reliability     =   QoSReliabilityPolicy.BEST_EFFORT,
            durability      =   QoSDurabilityPolicy.VOLATILE,
            history         =   QoSHistoryPolicy.KEEP_LAST,
            depth           =   1
        )

        # define number of drones
        self.n_drone    =   1

        # define subscribers and publishers
        self.array_publishers     =   [{'offboard_mode_pub':None, 'trajectory_pub':None, 'vehicle_command_pub':None} for _ in range(self.n_drone)]
        self.array_subscribers    =   [{'status_sub':None, 'local_pos_sub':None, 'global_pos_sub':None} for _ in range(self.n_drone)]

        for idx in range(self.n_drone):

            if self.n_drone == 1:
                self.ns         =   f''
            else:
                self.ns             =   f'px4_{idx+1}'

            self.array_subscribers[idx]['status_sub']           =   self.create_subscription(
                VehicleStatus,
                f'{self.ns}/fmu/out/vehicle_status',
                partial(self.vehicle_status_callback,id=idx),                           # instead of lambda function, lambda msg: self.vehicle_status_callback(msg,id=i), use partial function
                qos_profile_sub)

            self.array_subscribers[idx]['local_pos_sub']        =   self.create_subscription(
                VehicleLocalPosition,
                f'{self.ns}/fmu/out/vehicle_local_position',
                partial(self.local_position_callback,id=idx),
                qos_profile_sub)

            self.array_subscribers[idx]['global_pos_sub']       =   self.create_subscription(
                VehicleGlobalPosition,
                f'{self.ns}/fmu/out/vehicle_global_position',
                partial(self.global_position_callback,id=idx),
                qos_profile_sub)

            self.array_publishers[idx]['offboard_mode_pub']     =   self.create_publisher(
                OffboardControlMode,
                f'{self.ns}/fmu/in/offboard_control_mode',
                qos_profile_pub)

            self.array_publishers[idx]['trajectory_pub']        =   self.create_publisher(
                TrajectorySetpoint,
                f'{self.ns}/fmu/in/trajectory_setpoint',
                qos_profile_pub)

            self.array_publishers[idx]['vehicle_command_pub']   =   self.create_publisher(
                VehicleCommand,
                f'{self.ns}/fmu/in/vehicle_command',
                qos_profile_pub)
                   
        # parameters for callback
        self.timer_period   =   np.float64(0.02)                                            # [sec] callback function frequency (offboard mode should be at least 2Hz)
        self.timer          =   self.create_timer(self.timer_period, self.cmdloop_callback) # [-] command callback loop timer

        # get amplitude/frequency/phase/duration infromation
        self.amplitude      =   amplitude
        self.frequency      =   frequency
        self.phase          =   phase
        self.count_duration =   np.uint32(duration/self.timer_period)

        # variables for agents
        self.entry_execute      =   [False for _ in range(self.n_drone)]
        self.flight_phase       =   np.uint8(0)
        self.nav_state_list     =   [VehicleStatus.NAVIGATION_STATE_MAX for _ in range(self.n_drone)]
        self.next_phase_flag    =   False

        self.arm_counter_list       =   [0 for i in range(self.n_drone)]
        self.offboard_counter_list  =   [np.uint32(0) for i in range(self.n_drone)]
        self.local_pos_ned_list     =   [None for _ in range(self.n_drone)]
        self.global_ref_lla_list    =   [None for _ in range(self.n_drone)]

        self.trajectory_set_pt      =   []
        self.yaw_set_pt             =   []
        for i in range(self.n_drone):
            self.trajectory_set_pt.append(np.array([0,0,0], dtype=np.float64))
            self.yaw_set_pt.append(np.array(0.0, dtype=np.float64))


    # subscriber callback
    def vehicle_status_callback(self,msg,id):
        self.nav_state_list[id] = msg.nav_state

    def local_position_callback(self,msg,id):
        self.local_pos_ned_list[id]     =   np.array([msg.x,msg.y,msg.z], dtype=np.float64)

    def global_position_callback(self,msg,id):
        self.global_ref_lla_list[id]    =   np.array([msg.lat,msg.lon,msg.alt], dtype=np.float64)

    def publish_vehicle_command(self,command,id,param1=0.0,param2=0.0,param3=0.0):
        msg                     =   VehicleCommand()
        msg.param1              =   param1
        msg.param2              =   param2
        msg.param3              =   param3
        msg.command             =   command     # command ID
        msg.target_system       =   0           # system which should execute the command
        msg.target_component    =   1           # component which should execute the command, 0 for all components
        msg.source_system       =   1           # system sending the command
        msg.source_component    =   1           # component sending the command
        msg.from_external       =   True
        msg.timestamp           =   int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.array_publishers[id]['vehicle_command_pub'].publish(msg)

    def publish_offboard_control_mode(self,id):
        msg                     =   OffboardControlMode()
        msg.timestamp           =   int(Clock().now().nanoseconds/1000)
        msg.position            =   True
        msg.velocity            =   False
        msg.acceleration        =   False
        msg.attitude            =   False
        msg.body_rate           =   False
        self.array_publishers[id]['offboard_mode_pub'].publish(msg)

    def publish_trajectory_setpoint(self,id):
        msg                     =   TrajectorySetpoint()
        msg.timestamp           =   int(Clock().now().nanoseconds/1000)
        msg.position            =   np.array(self.trajectory_set_pt[id], dtype=np.float32)
        msg.yaw                 =   float(self.yaw_set_pt[id])
        self.array_publishers[id]['trajectory_pub'].publish(msg)

    def cmdloop_callback(self):

        # flight phase 0: idle/arming/takeoff
        if self.flight_phase == 0:

            # entry: 
            for idx in (idx for idx in range(self.n_drone) if self.entry_execute[idx] is False):
                # set up trajectory setpoint to begin offboard control mode
                if self.global_ref_lla_list[idx] is not None and self.local_pos_ned_list[idx] is not None and self.entry_execute[idx] is False:
                    self.trajectory_set_pt[idx] =   np.array([0.0,0.0,-10.0], dtype=np.float64)
                    self.yaw_set_pt[idx]        =   self.yaw_set_pt[idx]
                    self.publish_trajectory_setpoint(idx)
                    self.entry_execute[idx]     =   True

            # during:
            for idx in range(self.n_drone):
                if self.nav_state_list[idx] != VehicleStatus.ARMING_STATE_ARMED or self.arm_counter_list[idx] < 20:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,idx,1.0,6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,idx,1.0)
                    self.arm_counter_list[idx]  +=  1
                    self.get_logger().info('Drone #'+str(idx+1)+' armed and dangerous....')

            for idx in (idx for idx in range(self.n_drone) if self.entry_execute[idx] is True):
                self.publish_offboard_control_mode(idx)
                self.publish_trajectory_setpoint(idx)

            # exit:
            if [True for idx in range(self.n_drone) if (self.local_pos_ned_list[idx] is not None)] == \
                [True for idx in range(self.n_drone)]:
                if [True for idx in range(self.n_drone) if (np.linalg.norm(self.trajectory_set_pt[idx]-self.local_pos_ned_list[idx]) < 1.0)] == \
                    [True for idx in range(self.n_drone)]:
                    self.next_phase_flag        =   True 


        elif self.flight_phase == 1:
                
            # entry:
            if all(not agent_entry for agent_entry in self.entry_execute):
                for idx in range(self.n_drone):
                    self.offboard_counter_list[idx] =   np.uint32(0)
                    self.entry_execute[idx]         =   True
 
            # during:
            for idx in (idx for idx in range(self.n_drone) if self.entry_execute[idx] is True):

                self.trajectory_set_pt[idx] =   np.array([self.amplitude[0]*np.sin(2*np.pi*((self.frequency[0,0]-self.frequency[0,1])/self.count_duration*self.offboard_counter_list[idx]
                                                                                            +self.frequency[0,0])*self.offboard_counter_list[idx]*self.timer_period+self.phase[0]),
                                                          self.amplitude[1]*np.sin(2*np.pi*((self.frequency[1,0]-self.frequency[1,1])/self.count_duration*self.offboard_counter_list[idx]
                                                                                            +self.frequency[1,0])*self.offboard_counter_list[idx]*self.timer_period+self.phase[1]),
                                                          self.amplitude[2]*np.sin(2*np.pi*((self.frequency[2,0]-self.frequency[2,1])/self.count_duration*self.offboard_counter_list[idx]
                                                                                            +self.frequency[2,0])*self.offboard_counter_list[idx]*self.timer_period+self.phase[2])-10.0],
                                                          dtype=np.float64)
                
                self.yaw_set_pt[idx]        =   2*np.pi*((self.frequency[3,0]-self.frequency[3,1])/
                                                         self.count_duration*self.offboard_counter_list[idx]+self.frequency[3,0])*self.offboard_counter_list[idx]*self.timer_period+self.phase[3]

                self.publish_offboard_control_mode(idx)
                self.publish_trajectory_setpoint(idx)
                self.offboard_counter_list[idx] += 1
                self.get_logger().info('Drone #'+str(idx+1)+' offboard position control mode active with count: '+ str(self.offboard_counter_list[idx]))

            # exit:
            if [True for idx in range(self.n_drone) if (self.offboard_counter_list[idx] >= self.count_duration)] == \
                [True for idx in range(self.n_drone)]:
                self.next_phase_flag        =   True

        else:

            # entry:
            for idx in (idx for idx in range(self.n_drone) if self.entry_execute[idx] is False):

                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,idx,1.0,4.0,5.0)
                self.entry_execute[idx]     =   True

        if self.next_phase_flag:

            past_flag_temp          =   self.flight_phase
            self.flight_phase       =   self.flight_phase+1
            new_flag_temp           =   self.flight_phase
            self.next_phase_flag    =   False
            self.entry_execute      =   [False for _ in range(self.n_drone)]

            print('Next Flight Phase %d -> %d' %(past_flag_temp,new_flag_temp))

def main():

    # to be updated: get argument as a separate note for dynamically changing network (ndrones, ref_lla, wpts, formation)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', type=int)
    # args = parser.parse_args()

    amplitude       =   np.array([5.0, 5.0, 5.0, 1.0], dtype=np.float64)
    frequency       =   np.array([[1.0, 1.25], [1.30, 1.50], [1.70, 1.40], [0.10, 0.20]], dtype=np.float64)
    phase           =   np.array([0.0, np.pi/6, np.pi/7, 0.0])
    duration        =   200

    rclpy.init(args=None)

    offboard_pos_ctrl = OffboardPositionLissajous(amplitude, frequency, phase, duration)

    rclpy.spin(offboard_pos_ctrl)

    offboard_pos_ctrl.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()