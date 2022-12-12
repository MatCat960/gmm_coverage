cmake_minimum_required(VERSION 3.5)
project(gmm_coverage)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(turtlebot3_msgs REQUIRED)
find_package(GaussianMixtureModel REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(SFML REQUIRED COMPONENTS graphics window system)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(${PROJECT_NAME}_lib
 "src/distributed_gmm.cpp"
 "src/supervisor_gmm.cpp"
  "src/centralized_gmm.cpp"
  "src/gmm_visualizer.cpp"
  "src/hs_interface.cpp"
)


set(dependencies
  "geometry_msgs"
  "nav_msgs"
  "rclcpp"
  "rclpy"
  "tf2"
  "tf2_ros"
  "sensor_msgs"
  "turtlebot3_msgs"
  "visualization_msgs"
  "GaussianMixtureModel"
)

  set(SFML_libraries
  "sfml-graphics"
  "sfml-window"
  "sfml-system"
)

target_link_libraries(${PROJECT_NAME}_lib)
ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})

add_executable(hs_interface src/hs_interface.cpp )
target_link_libraries(hs_interface ${SFML_libraries})
ament_target_dependencies(hs_interface ${dependencies})

add_executable(centralized_gmm src/centralized_gmm.cpp )
target_link_libraries(centralized_gmm ${SFML_libraries})
ament_target_dependencies(centralized_gmm ${dependencies})

add_executable(distributed_gmm src/distributed_gmm.cpp )
target_link_libraries(distributed_gmm ${SFML_libraries})
ament_target_dependencies(distributed_gmm ${dependencies})

add_executable(gmm_visualizer src/gmm_visualizer.cpp )
target_link_libraries(gmm_visualizer ${SFML_libraries})
ament_target_dependencies(gmm_visualizer ${dependencies})

add_executable(supervisor_gmm src/supervisor_gmm.cpp )
target_link_libraries(supervisor_gmm ${SFML_libraries})
ament_target_dependencies(supervisor_gmm ${dependencies})


# Install python modules
ament_python_install_package(${PROJECT_NAME})

# Install python executables
install(PROGRAMS
  scripts/interface.py
  DESTINATION lib/${PROJECT_NAME}
  )

  
install(TARGETS hs_interface centralized_gmm distributed_gmm gmm_visualizer 
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/gmm_coverage
  DESTINATION include
)

# let the compiler search for headers in include folder
include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(geometry_msgs)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(rclcpp)
ament_export_dependencies(rclpy)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(tf2)
ament_export_dependencies(tf2_ros)
ament_export_dependencies(turtlebot3_msgs)
ament_export_dependencies(visualization_msgs)
ament_export_dependencies(GaussianMixtureModel)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
