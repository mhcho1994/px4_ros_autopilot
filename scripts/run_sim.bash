#!/bin/bash
#set -e

# Parse help argument
if [[ ${1} == "-h" ]] || [[ ${1} == "--help" ]]; then
  echo -e "Usage: Run_PX4_base.bash [option...]"
  echo -e "This script builds & runs PX4_Autopilot and ROS 2 mircroRTPS Bridge" 
  echo 
  echo -e "\t--PX4_Autopilot_path \t Location of the PX4 Autopilot workspace where one has cloned PX4_Autopilot. Default: $HOME/PX4_Autopilot"
  echo -e "\t--QGroundControl_path \t Location of QGroundControl application where one has cloned QGroundControl.AppImage. Default: $HOME"
  echo -e "\t--px4_ros_ws_path \t Set ROS (2) working space location. If not set, the script sources the environment in $HOME/px4_ros_dev"
  echo -e "\t--build_done \t\t Select whether the build of the ROS (2) side FastRTPS/DDS bridge is completed. If not set, the default value is True"
  echo -e "\t--px4_ros_com_path \t Set ROS (2) side FastRTPS/DDS bridge repository location. If not set, the script builds the repository in $HOME/px4_ros_dev"
  echo -e "\t--px4_msgs_path \t\t Set px4_msgs repository location. If not set, the script builds the repository in $HOME/px4_ros_dev"
  echo -e "\t--verbose \t\t Add more verbosity to the console output"
  echo >&2
fi

# Figure out the absolute path of this script
SCRIPT_DIR=${0}
if [[ ${SCRIPT_DIR:0:1} != '/' ]]; then
  SCRIPT_DIR=$(dirname $(realpath -s "${PWD}/${0}"))
fi

# Parse the arguments
while [ $# -gt 0 ]; do
  if [[ ${1} == *"--"* ]]; then
    v="${1/--/}"
    if [ ! -z ${2} ]; then
      if [ ${2:0:2} != '--' ]; then
        declare ${v}="${2}"
      fi
    fi
  fi
  shift
done

# Check if specific terminal exists
# gnome-terminal
if [ -x "$(command -v gnome-terminal)" ]; then
  SHELL_TERM="gnome-terminal --tab -- /bin/bash -c"
  echo "Terminal: ${SHELL_TERM}"
# xterm
elif [ -x "$(command -v xterm)" ]; then
  SHELL_TERM="xterm -e"
  echo "Terminal: ${SHELL_TERM}"
fi

# Build and run PX4-Autopilot Gazebo simulation
echo "============ Run PX4-Gazebo simulation ============"
PX4_SIM_MODE="sitl_rtps"
PX4_SIMULATOR="gazebo"
if [ -z ${PX4_Autopilot_path} ]; then
  echo -e "The PX4 source code location is not specified"
  echo -e "Check the default directory ..."
  PX4_Autopilot_path="${HOME}/PX4-Autopilot"
  echo ''${PX4_Autopilot_path}''
  if [ -e ${PX4_Autopilot_path} ]; then
    echo -e "PX4-Autopilot source exists in the default directory"
  else
    echo -e "PX4-Autopilot source is not installed in the default directory, " #please download it using git clone https://github.com/PX4/PX4-Autopilot.git --recursive
  fi
else
  echo -e "Check the given directory ..."
  if [ -e ${PX4_Autopilot_path} ]; then
    echo -e "PX4-Autopilot source exists in the given directory"
  else
    echo -e "PX4-Autopilot source is not in the specified directory, " #please set the correct working space direcotry with '--PX4_Autopilot_path' argument (ex: ~/)
  fi
fi

PX4_PATH=${PX4_Autopilot_path}
unset PX4_Autopilot_path

${SHELL_TERM} \
  '''
  # Run PX4 simulation in a newly opened terminal 
  'cd\ ${PX4_PATH}'
  'sleep\ 1'
  'make\ px4_${PX4_SIM_MODE}\ ${PX4_SIMULATOR}'
  'exit'
  ''' &

# Run QGroundControl Application in a new terminal window
echo "============ Run QGroundControl.AppImage ============"
if [ -z ${QGroundControl_path} ]; then
  echo -e "The location of QGroundControl.AppImage is not specified"
  echo -e "Check the default path ..."
  QGroundControl_path=${HOME}
  if [ -e "${HOME}/QGroundControl.AppImage" ]; then
    echo -e "QGC exists in the default directory"
  else
    echo -e "QGC is not installed in the default directory, please download it from http://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html"
  fi
else
  echo -e "Check the given directory ..."
  if [ -e "${QGroundControl_path}/QGroundControl.AppImage" ]; then
    echo -e "QGC exists in the given directory"
  else
    echo -e "QGC is not installed in the specified directory, please set the correct install location with '--QGroundControl_path' argument (ex: ~/QGroundControl)"
  fi
fi

QGC_PATH=${QGroundControl_path}
unset QGroundControl_path

$SHELL_TERM \
  '''
  # Run QGroundControl.AppImage in a newly opened terminal
  'cd\ ${QGC_PATH}'
  './QGroundControl.AppImage'
  'exit'
  ''' &

# Build and run Run ROS (2) side FastRTPS/DDS bridge in a new terminal window
echo "============ Build & Source ROS packages ============"
if [ -z ${px4_ros_ws_path} ]; then
  echo -e "The location of ROS working space is not specified"
  echo -e "Check the default directory ..."
  px4_ros_ws_path="${HOME}/px4_ros_autopilot"
  if [ -e ${px4_ros_ws_path} ]; then
    echo -e "ROS working space exists in the default directory"
  else
    echo -e "ROS working space is not in the default directory, please set the working space direcotry with '--px4_ros_ws_path' argument (ex: ~/)"
  fi
else
  echo -e "Check the given directory ..."
  if [ -e ${px4_ros_ws_path} ]; then
    echo -e "ROS working space exists in the given directory"
  else
    echo -e "ROS working space is not in the specified directory, please set the correct working space direcotry with '--px4_ros_ws_path' argument (ex: ~/)"
  fi
fi

ROS_COM_SCRIPT_DIR="${px4_ros_ws_path}/install/setup.bash"
unset px4_ros_ws_path

if [ ! -v ${build_done} ]; then
  echo -e "The build status is not specified"
  echo -e "Check the source script ..."
  if [ -e ${ROS_COM_SCRIPT_DIR} ]; then
    echo -e "The ROS (2) side FastRTPS/DDS bridge is confirmed"
    build_done=True
  else
    echo -e "The ROS (2) side FastRTPS/DDS bridge is not built"
    build_done=False
  fi
fi

if [ ! ${build_done} ]; then
  echo -e "Need to be built - this function will be added in the future release"
fi

BUILD_OK=${build_done}
unset build_done

$SHELL_TERM \
  '''
  # Run ROS (2) side FastRTPS/DDS bridge in a newly opened terminal
  'source\ ${ROS_COM_SCRIPT_DIR}'
  'micrortps_agent\ -t\ UDP'
  'exit'
  ''' &

unset SCRIPT_DIR
unset SHELL_TERM

unset PX4_PATH
unset PX4_SIM_MODE
unset PX4_SIMULATOR

unset QGC_PATH

unset ROS_COM_SCRIPT_DIR
unset BUILD_OK