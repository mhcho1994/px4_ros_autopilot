# Load public libraries
import numpy
import math

# Load ROS2 related libraries

# Load ROS2 messages

class controller():

    # Define controller module parameters
    def load_parameters(self):

        # Controller (attitude commander) time step related parameters
        self.ctrl_freq_             =   numpy.float32(100.0)                                        # [Hz] Autopilot frequency
        self.ctrl_dt_               =   1/self.ctrl_freq_                                           # [sec] Autopilot time step
        self.persistent_timer_      =   numpy.float32(0.1)                                          # [sec] Persistent timer counter threshold

        # Physical properties/constants
        self.quadrotor_mass         =   numpy.float32(1.50+0.01*0)                                  # [kg] Mass of quadrotor
        self.gravity_const          =   numpy.float32(9.8066)                                       # [m/s^2] Gravitational acceleration

class autopilot():

    # Define autopilot module parameters
    def load_parameters(self):

        # Autopilot (guidance law) time step related parameters
        self.ap_freq_               =   numpy.float32(100.0)                                        # [Hz] Autopilot frequency
        self.ap_dt_                 =   1/self.ap_freq_                                             # [sec] Autopilot time step
        self.persistent_timer_      =   numpy.float32(0.1)                                          # [sec] Persistent timer counter threshold

        # Guidance law parameters
        self.eta_                   =   numpy.float32(2.0)                                          # [-] Autopilot gain
        self.T_horizon_             =   numpy.float32(3.0)                                          # [sec] Time horizon for virtual target
        self.LA_dist_               =   numpy.float32(0.0)                                          # [m] Look-Ahead distance -> Do not initialize here

        self.V_desired_             =   numpy.float32(5.0)                                          # [m/s] Desired ground speed
        self.V_lbound_              =   numpy.float32(0.1)                                          # [m/s] Lowest bound of speed
        self.eps_                   =   1.0e-3                                                      # [-] Constant preventing ill-condition

        # Physical constants
        self.gravity_const_         =   numpy.float32(9.8066)                                       # [m/s^2] Gravitational acceleration

        # Initialize parameters for heading error guidance (Kinematic variables)
        self.R_rel_                 =   numpy.zeros(3,dtype=numpy.float32)                          # [m] Relative position from vehicle to virtual target
        self.V_rel_                 =   numpy.empty(3,dtype=numpy.float32)                          # [m/s] Relative velocity between vehicle and virtual target
        self.R_rel_norm_            =   numpy.empty(1,dtype=numpy.float32)                          # [m] Relative range from vehicle to virtual target
        self.V_UAV_norm_            =   numpy.empty(1,dtype=numpy.float32)                          # [m/s] Speed of vehicle
        self.Vclose_                =   numpy.empty(1,dtype=numpy.float32)                          # [m/s] Close speed
        self.Tgo_                   =   numpy.empty(1,dtype=numpy.float32)                          # [sec] Time-to-go
        self.Lambda_                =   numpy.empty(2,dtype=numpy.float32)                          # [rad] Heading/climb angle to virtual target (line of sight)
        self.lead_angle_            =   numpy.empty(2,dtype=numpy.float32)                          # [rad] Lead angle to virtual target (velocity vector and line of sight)

        # Initialize states of virtual target
        self.X_virtual_             =   numpy.zeros(6,dtype=numpy.float32)                          # [m,m/s,rad] Virtual target position, speed, heading and climb angle

        # Initialize parameters and states of velocity controller
        self.Kp_vel_                =   numpy.float32(0.4)
        self.Ki_vel_                =   numpy.float32(self.Kp_vel_*0.00)
        self.vel_err_               =   numpy.float32(0.0)
        self.int_vel_err_           =   numpy.float32(0.0)

class wpt_manager():

    # Define waypoint module parameters
    def load_parameters(self):

        # Waypoint manager time step related parameters
        self.wpt_freq_              =   numpy.float32(100.0)                                        # [Hz] Waypoint manager frequency
        self.wpt_dt_                =   1/self.wpt_freq_                                            # [sec] Waypoint manager time step
        self.persistent_timer_      =   numpy.float32(0.04)                                         # [sec] Persistent timer counter threshold

# Define required utility function
def get_3D_to_AzEl(vec):

    # Get azimuth and elevation from 3-dimensional vector
    psi         =   math.atan2(vec[1],vec[0])
    the         =   math.atan2(-vec[2],math.sqrt(pow(vec[0],2)+pow(vec[1],2)))

    AzEl        =   numpy.empty(2)
    AzEl[0]     =   psi
    AzEl[1]     =   the

    return AzEl

def get_AzEl_to_3D(angle):

    # Convert azimuth and elevation to 3-dimensional unit vector
    psi     =   angle[0]
    the     =   angle[1]

    cpsi    =   math.cos(psi)
    spsi    =   math.sin(psi)
    cthe    =   math.cos(the)
    sthe    =   math.sin(the)

    u       =   numpy.empty(3)
    u[0]    =   cthe*cpsi
    u[1]    =   cthe*spsi
    u[2]    =   -sthe

    return u

def get_Transform_3D(angle):

    # Get coordinate transformation matrix between two coordinates using azimuth and elevation
    psi     =   angle[0]
    the     =   angle[1]

    cpsi    =   math.cos(psi)
    spsi    =   math.sin(psi)
    cthe    =   math.cos(the)
    sthe    =   math.sin(the)

    DCM_psi         =   numpy.zeros((3,3))
    DCM_psi[0,0]    =   cpsi
    DCM_psi[0,1]    =   spsi
    DCM_psi[1,0]    =   -spsi
    DCM_psi[1,1]    =   cpsi
    DCM_psi[2,2]    =   1

    DCM_the         =   numpy.zeros((3,3))
    DCM_the[0,0]    =   cthe
    DCM_the[0,2]    =   -sthe
    DCM_the[1,1]    =   1
    DCM_the[2,0]    =   sthe
    DCM_the[2,2]    =   cthe

    return numpy.matmul(DCM_the,DCM_psi)

def Integrate_Euler(dydx,y,dt):

    return y+dydx*dt