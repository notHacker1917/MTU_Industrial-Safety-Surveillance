# VisionDetection.msg
# Custom message for detailed vision detection outputs
# Publish on /vision/detections_typed for structured data

int32 track_id
float32[4] bbox
float32 depth_m
bool suit
bool shield
bool gloves
bool compliant
string alert_level
string zone
float32[2] centroid
