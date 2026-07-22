"""
ROS2 Launch File for Vision Node
Launches the VisionNode with configurable parameters and optional mock mode
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    Generate ROS2 launch description for vision node.

    Returns:
        LaunchDescription with all nodes and configurations
    """

    # Declare launch arguments
    blob_path_arg = DeclareLaunchArgument(
        "blob_path",
        default_value="./yolov26n_320_320.blob",
        description="Path to YOLO26n.blob model file",
    )

    tflite_model_path_arg = DeclareLaunchArgument(
        "tflite_model_path",
        default_value="./model.tflite",
        description="Path to PPE TFLite model file",
    )

    ppe_conf_threshold_arg = DeclareLaunchArgument(
        "ppe_conf_threshold",
        default_value="0.65",
        description="PPE classification confidence threshold (0.0-1.0)",
    )

    mock_mode_arg = DeclareLaunchArgument(
        "mock_mode",
        default_value="false",
        description="Use mock mode (webcam) instead of OAK-D",
    )

    # Vision node
    vision_node = Node(
        package="",  # No package specified; assumes running from directory
        executable="ros2_vision_node.py",
        name="vision_node",
        output="screen",
        parameters=[
            {"blob_path": LaunchConfiguration("blob_path")},
            {"tflite_model_path": LaunchConfiguration("tflite_model_path")},
            {"ppe_conf_threshold": LaunchConfiguration("ppe_conf_threshold")},
            {"mock_mode": LaunchConfiguration("mock_mode")},
        ],
    )

    # Log configuration
    log_config = LogInfo(
        msg=[
            "Starting Vision Node with configuration:",
            " - blob_path: ",
            LaunchConfiguration("blob_path"),
            " - tflite_model_path: ",
            LaunchConfiguration("tflite_model_path"),
            " - ppe_conf_threshold: ",
            LaunchConfiguration("ppe_conf_threshold"),
            " - mock_mode: ",
            LaunchConfiguration("mock_mode"),
        ]
    )

    # Return launch description
    return LaunchDescription(
        [
            blob_path_arg,
            tflite_model_path_arg,
            ppe_conf_threshold_arg,
            mock_mode_arg,
            log_config,
            vision_node,
        ]
    )
