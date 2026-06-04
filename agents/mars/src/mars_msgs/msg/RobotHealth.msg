## mars_msgs/msg/RobotHealth
##
## Self-published by the sim's health / fault-injection node.
## The Aggregator combines it with sensor_msgs/BatteryState to set fault_flag.
## Publishing a fault here is the primary mechanism for triggering demos.

std_msgs/Header header

uint8 LEVEL_OK=0
uint8 LEVEL_WARN=1
uint8 LEVEL_ERROR=2

string    robot_id
uint8     level          # overall health level (use constants above)
bool      estop_active
string[]  fault_codes    # e.g. ["MOTOR_FAULT","LIDAR_TIMEOUT"]
