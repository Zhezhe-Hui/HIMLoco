cmake_minimum_required(VERSION 3.8)
project(bigdog_urdf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 仅保留编译期必需依赖
find_package(ament_cmake REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(urdf REQUIRED)

# 按原目录结构安装文件（核心修复！）
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)
install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(DIRECTORY config-back/
  DESTINATION share/${PROJECT_NAME}/config-back
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
