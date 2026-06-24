cmake_minimum_required(VERSION 3.8)
project(iw_hub_description)

find_package(ament_cmake REQUIRED)

# urdf / launch / rviz 디렉토리를 install 공간에 복사
install(
  DIRECTORY urdf launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
