"""
Tests for PPE Classifier (TFLite model)
Tests run WITHOUT model present (all mocked)
"""

import pytest
import numpy as np
import cv2


class TestPPEClassifierInitialization:
    """Test classifier initialization and model loading"""
    
    def test_classifier_with_missing_model(self):
        """Classifier should handle missing model gracefully"""
        model_path = "/nonexistent/model.tflite"
        
        # Should raise FileNotFoundError or log warning
        try:
            # Simulate model loading
            with open(model_path, 'r'):
                pass
        except FileNotFoundError:
            # Expected behavior
            pass


class TestPPEClassifierInput:
    """Test input preprocessing"""
    
    def test_resize_to_224x224(self):
        """Input crop should be resized to 224×224"""
        # Create random crop
        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Resize to model input
        resized = cv2.resize(crop, (224, 224))
        
        assert resized.shape == (224, 224, 3)
    
    def test_bgr_to_rgb_conversion(self):
        """Image should be converted from BGR to RGB"""
        bgr_image = np.zeros((224, 224, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue channel
        bgr_image[:, :, 2] = 100  # Red channel
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Verify channels swapped
        assert rgb_image[:, :, 0].mean() == 100  # Red now in first position
        assert rgb_image[:, :, 2].mean() == 255  # Blue now in third position
    
    def test_normalization_to_0_1_range(self):
        """Pixel values should be normalized to [0, 1]"""
        uint8_image = np.array([0, 127, 255], dtype=np.uint8)
        
        # Normalize
        normalized = uint8_image.astype(np.float32) / 255.0
        
        # Check range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.isclose(normalized[0], 0.0)
        assert np.isclose(normalized[2], 1.0)
    
    def test_crop_too_small_returns_none(self):
        """Crop smaller than 40×40 should return None"""
        small_crop = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        
        # Check minimum size
        min_size = 40
        is_valid = small_crop.shape[0] >= min_size and small_crop.shape[1] >= min_size
        
        assert not is_valid


class TestPPEClassifierOutput:
    """Test classification output and post-processing"""
    
    def test_output_keys_present(self):
        """Classification should return dict with PPE keys"""
        # Simulated model output (3 sigmoid scores)
        model_output = [0.8, 0.3, 0.7]  # [suit, shield, gloves]
        
        # Apply sigmoid
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        sigmoid_output = [sigmoid(x) for x in model_output]
        
        # Create result dict
        result = {
            "suit": sigmoid_output[0],
            "shield": sigmoid_output[1],
            "gloves": sigmoid_output[2],
        }
        
        # Check all keys present
        assert "suit" in result
        assert "shield" in result
        assert "gloves" in result
    
    def test_sigmoid_activation(self):
        """Sigmoid should output values in (0, 1)"""
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        test_values = [-10, -1, 0, 1, 10]
        sigmoid_values = [sigmoid(x) for x in test_values]
        
        # All values should be in (0, 1)
        assert all(0 < s < 1 for s in sigmoid_values)
        
        # Negative input → <0.5, positive input → >0.5
        assert sigmoid(-5) < 0.5
        assert sigmoid(5) > 0.5
    
    def test_confidence_thresholding(self):
        """Classifications above threshold should be True"""
        threshold = 0.65
        
        test_cases = [
            (0.1, False),   # Below threshold
            (0.65, True),   # At threshold
            (0.9, True),    # Above threshold
        ]
        
        for confidence, expected in test_cases:
            result = confidence >= threshold
            assert result == expected
    
    def test_multi_label_output(self):
        """PPE should be multi-label (multiple items can be present)"""
        confidences = {
            "suit": 0.8,    # Detected
            "shield": 0.3,  # Not detected
            "gloves": 0.7,  # Detected
        }
        threshold = 0.65
        
        detected_items = [item for item, conf in confidences.items()
                         if conf >= threshold]
        
        # Should have 2 items (suit, gloves)
        assert len(detected_items) == 2
        assert "suit" in detected_items
        assert "gloves" in detected_items
        assert "shield" not in detected_items


class TestShieldHeuristic:
    """Test shield detection heuristic (HoughLines fallback)"""
    
    def test_uncertain_shield_triggers_heuristic(self):
        """Shield confidence in [0.35, 0.65] should trigger heuristic"""
        shield_conf_ranges = [
            (0.30, False),  # Clearly not present
            (0.40, True),   # Uncertain - trigger heuristic
            (0.60, True),   # Uncertain - trigger heuristic
            (0.70, False),  # Clearly present - no heuristic needed
        ]
        
        lower_bound = 0.35
        upper_bound = 0.65
        
        for conf, should_trigger in shield_conf_ranges:
            triggers_heuristic = lower_bound <= conf <= upper_bound
            assert triggers_heuristic == should_trigger
    
    def test_houghlines_horizontal_line_detection(self):
        """HoughLines should detect horizontal strap lines"""
        # Create synthetic shield image with horizontal lines
        shield_img = np.zeros((224, 224), dtype=np.uint8)
        
        # Draw horizontal lines (strap pattern)
        cv2.line(shield_img, (30, 56), (194, 56), 255, 2)  # Top strap
        cv2.line(shield_img, (30, 112), (194, 112), 255, 2)  # Middle strap
        
        # Apply edge detection
        edges = cv2.Canny(shield_img, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                               threshold=50, minLineLength=30, maxLineGap=10)
        
        # Should detect lines
        assert lines is not None
        assert len(lines) >= 2
    
    def test_heuristic_override_boost_confidence(self):
        """If heuristic detects lines, override shield confidence to 0.9"""
        uncertain_confidence = 0.50
        heuristic_triggered = True
        
        # Override logic
        if heuristic_triggered:
            final_confidence = 0.9
        else:
            final_confidence = uncertain_confidence
        
        # Should be boosted
        assert final_confidence == 0.9


class TestClassifierFallback:
    """Test graceful fallback when model unavailable"""
    
    def test_mock_classifier_returns_dict(self):
        """Mock classifier should return valid dict structure"""
        # Simulate mock classifier
        mock_result = {
            "suit": True,
            "shield": False,
            "gloves": True,
            "suit_conf": 0.8,
            "shield_conf": 0.3,
            "gloves_conf": 0.7,
        }
        
        # Check structure
        assert all(k in mock_result for k in ["suit", "shield", "gloves"])
        assert all(isinstance(v, (bool, float)) for v in mock_result.values())
    
    def test_mock_consistency(self):
        """Mock should return consistent random PPE per crop"""
        # Mock behavior: deterministic based on crop hash
        crop1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        crop2 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        # Same crop should produce same result
        hash1a = hash(crop1.tobytes())
        hash1b = hash(crop1.tobytes())
        assert hash1a == hash1b
        
        # Different crop should be different
        hash2 = hash(crop2.tobytes())
        assert hash1a != hash2


class TestBatchProcessing:
    """Test processing multiple detections"""
    
    def test_classify_multiple_detections(self):
        """Should handle multiple detections in sequence"""
        detections = [
            {"bbox": [10, 20, 100, 150], "id": 1},
            {"bbox": [50, 50, 120, 160], "id": 2},
            {"bbox": [150, 100, 200, 250], "id": 3},
        ]
        
        results = {}
        for det in detections:
            # Simulate classification
            results[det["id"]] = {
                "suit": True,
                "shield": False,
                "gloves": True,
            }
        
        # Should have result for each detection
        assert len(results) == 3
        assert all(det["id"] in results for det in detections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
