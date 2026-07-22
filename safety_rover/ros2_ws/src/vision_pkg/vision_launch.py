"""
ROS2 launch file for Vision Stack
Launches OAK-D pipeline + ByteTrack + PPE classification + RosBridge integration
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate ROS2 launch description"""
    
    # Declare launch arguments
    blob_path_arg = DeclareLaunchArgument(
        'blob_path',
        default_value='models/yolov26n_320_320.blob',
        description='Path to YOLO blob model'
    )
    
    tflite_model_path_arg = DeclareLaunchArgument(
        'tflite_model_path',
        default_value='models/ppe_classifier_model.tflite',
        description='Path to TFLite PPE classifier model'
    )
    
    # Get launch configuration values
    blob_path = LaunchConfiguration('blob_path')
    tflite_model_path = LaunchConfiguration('tflite_model_path')
    
    # Vision node
    vision_node = Node(
        package='vision_pkg',
        executable='vision_node',
        name='vision_node',
        parameters=[{
            'blob_path': blob_path,
            'tflite_model_path': tflite_model_path,
            'frame_width': 320,
            'frame_height': 320,
            'detection_confidence': 0.45,
            'nms_threshold': 0.5,
            'bytetrack_iou_threshold': 0.3,
            'ppe_confidence_threshold': 0.65,
        }],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        blob_path_arg,
        tflite_model_path_arg,
        vision_node,
    ])
