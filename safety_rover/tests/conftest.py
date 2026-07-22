"""
Pytest configuration for Safety Rover tests
Provides common fixtures and configurations
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_frame():
    """Fixture: 320×320 mock frame (uint8)"""
    import numpy as np
    return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Fixture: Sample detection dict"""
    return {
        "bbox": [50, 50, 150, 200],
        "confidence": 0.85,
        "depth_m": 2.5,
        "class_id": 0,  # Person
    }


@pytest.fixture
def sample_detections():
    """Fixture: Multiple sample detections"""
    return [
        {"bbox": [10, 20, 100, 150], "confidence": 0.9, "depth_m": 2.0},
        {"bbox": [150, 50, 220, 180], "confidence": 0.75, "depth_m": 3.5},
        {"bbox": [50, 100, 130, 250], "confidence": 0.82, "depth_m": 1.8},
    ]


@pytest.fixture
def mock_tracker():
    """Fixture: Mock ByteTrack tracker state"""
    return {
        "tracks": [],
        "frame_id": 0,
        "max_time_lost": 30,
    }


@pytest.fixture
def mock_ppe_classifier():
    """Fixture: Mock PPE classifier"""
    class MockClassifier:
        def classify(self, crop):
            """Return mock PPE classification"""
            return {
                "suit": True,
                "shield": False,
                "gloves": True,
                "suit_conf": 0.8,
                "shield_conf": 0.3,
                "gloves_conf": 0.7,
            }
    return MockClassifier()


@pytest.fixture
def mock_pipeline():
    """Fixture: Mock OAK pipeline"""
    class MockPipeline:
        def __init__(self):
            self.frame_width = 320
            self.frame_height = 320
        
        def get_frame(self):
            """Return mock annotated frame + detections"""
            import numpy as np
            frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            detections = [
                {"bbox": [50, 50, 150, 200], "confidence": 0.85, "depth_m": 2.5},
            ]
            return frame, detections
    
    return MockPipeline()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests as requiring hardware (skip with '-m \"not hardware\"')"
    )


# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests in specific modules
        if "tracker_ppe" in str(item.fspath):
            item.add_marker(pytest.mark.tracking)
        elif "oak_pipeline" in str(item.fspath):
            item.add_marker(pytest.mark.vision)
        elif "ppe_classifier" in str(item.fspath):
            item.add_marker(pytest.mark.classification)
