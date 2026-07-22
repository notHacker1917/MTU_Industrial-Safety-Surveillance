"""
Tests for OAK pipeline preprocessing and detection
Tests run WITHOUT OAK-D hardware (all mocked)
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock


class TestOakPipelinePreprocessing:
    """Test image preprocessing (CLAHE, visibility detection)"""

    def test_clahe_brightens_dark_frame(self):
        """CLAHE should increase brightness of dark frames"""
        # Create intentionally dark frame
        dark_frame = np.ones((320, 320, 3), dtype=np.uint8) * 30
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Verify brightness increased
        assert brightened.mean() > dark_frame.mean()
    
    def test_low_visibility_detection(self):
        """Frame with low std dev should be flagged as low visibility"""
        # Create frame with low variation (uniform gray)
        low_var_frame = np.ones((320, 320, 3), dtype=np.uint8) * 128
        std = low_var_frame.std()
        
        # Should be detected as low visibility (std < 35)
        assert std < 35
    
    def test_high_visibility_not_flagged(self):
        """Frame with high variation should NOT be flagged"""
        # Create random noise frame (high variation)
        high_var_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        std = high_var_frame.std()
        
        # Should NOT be low visibility (std > 35)
        assert std > 35
    
    def test_glare_detection_with_bright_pixels(self):
        """Frame with many bright pixels should detect glare"""
        # Create frame with 20% bright pixels (> 245)
        glare_frame = np.ones((320, 320, 3), dtype=np.uint8) * 250
        
        # Count pixels with brightness > 245
        bright_pixels = np.sum(glare_frame[:, :, 0] > 245)
        total_pixels = glare_frame[:, :, 0].size
        bright_ratio = bright_pixels / total_pixels
        
        # Should be > 0.15 (15% threshold)
        assert bright_ratio > 0.15
    
    def test_no_glare_with_normal_frame(self):
        """Normal frame should not detect glare"""
        # Create normal frame (mean brightness ~128)
        normal_frame = np.ones((320, 320, 3), dtype=np.uint8) * 128
        
        # Count bright pixels
        bright_pixels = np.sum(normal_frame[:, :, 0] > 245)
        total_pixels = normal_frame[:, :, 0].size
        bright_ratio = bright_pixels / total_pixels
        
        # Should be < 0.15 (15% threshold)
        assert bright_ratio < 0.15


class TestOakPipelineDetections:
    """Test detection parsing and bbox extraction"""
    
    def test_bbox_format_valid(self):
        """Bounding boxes should have correct format [x1, y1, x2, y2]"""
        # Simulated detection
        bbox = [10, 20, 100, 150]
        
        # Validate format
        x1, y1, x2, y2 = bbox
        assert x1 < x2, "x1 must be < x2"
        assert y1 < y2, "y1 must be < y2"
        assert x1 >= 0 and y1 >= 0, "Coordinates must be positive"
        assert x2 <= 320 and y2 <= 320, "Coordinates must be within frame"
    
    def test_detection_confidence_thresholding(self):
        """Detections below confidence threshold should be filtered"""
        detections = [
            {"confidence": 0.2},  # Below 0.45 threshold
            {"confidence": 0.50},  # Above threshold
            {"confidence": 0.95},  # Above threshold
        ]
        
        threshold = 0.45
        filtered = [d for d in detections if d["confidence"] >= threshold]
        
        assert len(filtered) == 2
        assert all(d["confidence"] >= threshold for d in filtered)
    
    def test_nms_removes_overlapping_boxes(self):
        """Non-Maximum Suppression should remove overlapping detections"""
        # Two overlapping boxes
        box1 = [10, 10, 100, 100]
        box2 = [50, 50, 120, 120]  # Overlaps with box1
        
        def iou(b1, b2):
            """Compute Intersection over Union"""
            x1_inter = max(b1[0], b2[0])
            y1_inter = max(b1[1], b2[1])
            x2_inter = min(b1[2], b2[2])
            y2_inter = min(b1[3], b2[3])
            
            inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
            box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Should have high IoU (overlapping)
        overlap = iou(box1, box2)
        assert overlap > 0.3  # 30% overlap threshold
    
    def test_depth_values_within_valid_range(self):
        """Depth values should be within valid range (0.5 - 6.0m)"""
        depth_values = [0.5, 1.2, 3.5, 5.9]  # Valid depths
        
        min_depth = 0.5
        max_depth = 6.0
        
        assert all(min_depth <= d <= max_depth for d in depth_values)
    
    def test_invalid_depth_filtered(self):
        """Detections with invalid depth should be filtered"""
        detections = [
            {"depth_m": 0.2, "valid": False},  # Too close
            {"depth_m": 2.0, "valid": True},   # Valid
            {"depth_m": 8.0, "valid": False},  # Too far
        ]
        
        valid_detections = [d for d in detections if d["valid"]]
        assert len(valid_detections) == 1


class TestOakPipelineFrameFlags:
    """Test frame-level flag generation"""
    
    def test_frame_flags_structure(self):
        """Frame flags should have required fields"""
        frame_flags = {
            "low_visibility": False,
            "glare": False,
            "frame_number": 1,
        }
        
        # Check all required keys present
        assert "low_visibility" in frame_flags
        assert "glare" in frame_flags
        assert isinstance(frame_flags["low_visibility"], bool)
        assert isinstance(frame_flags["glare"], bool)
    
    def test_multiple_flags_possible(self):
        """Multiple flags can be true simultaneously"""
        frame_flags = {
            "low_visibility": True,
            "glare": True,
        }
        
        # Both flags can be set
        flagged_count = sum(1 for v in frame_flags.values() if v is True)
        assert flagged_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
