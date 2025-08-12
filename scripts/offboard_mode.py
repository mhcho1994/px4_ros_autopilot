#!/usr/bin/env python3

__author__ = "Minhyun Cho"
__contact__ = "@purdue.edu"

# python packages and modules
from enum import Enum

class OffboardControlType(Enum):
    POSITION        =  1
    VELOCITY        =  2
    ACCELERATION    =  3
    ATTITUDE        =  4
    RATE            =  5
    TORQUETHRUST    =  6
    ACTUATOR        =  7

class ControllerType(Enum):
    SE3_GEOMETRIC   =  1
    QUATERNION_PID  =  2