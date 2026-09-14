from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
# 新增：导入ParameterValue，显式声明参数类型
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_name = "bigdog_urdf"
    pkg_path = FindPackageShare(pkg_name)

    # 1. 声明URDF路径参数（恢复urdf/子目录）
    urdf_file_arg = DeclareLaunchArgument(
        "urdf_file",
        default_value=PathJoinSubstitution([pkg_path, "urdf", "bigdog.urdf"]),
        description="Path to URDF file"
    )

    # 2. 读取URDF内容 + 显式声明为字符串类型（核心修复！）
    robot_description = ParameterValue(
        Command(["cat ", LaunchConfiguration("urdf_file")]),
        value_type=str  # 告诉ROS2：这是字符串，不是YAML
    )

    # 3. 机器人状态发布器
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}]
    )

    # 4. 关节GUI
    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen"
    )

    # 5. RViz2（无config文件则注释arguments）
    rviz_config_path = PathJoinSubstitution([pkg_path, "config-back", "urdf_view.rviz"])
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path]
    )

    return LaunchDescription([
        urdf_file_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz2_node
    ])
