"""
TFLite PPE (Personal Protective Equipment) Classifier
Inference wrapper for INT8 MobileNetV3-Small multi-label classification model
Includes face shield heuristic fallback for transparent shields
"""

import cv2
import numpy as np
import time
import sys
import random
from typing import Dict, Optional, Any

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    Interpreter = None


class PPEClassifierMock:
    """
    FIX A8: Placeholder PPE classifier for demo testing before real model trained.
    Use during development with 70% simulated compliance rate.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.65) -> None:
        """Initialize placeholder classifier."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.compliant_rate = 0.70  # 70% compliance for realistic testing
        print(f"[PLACEHOLDER] Using PPE classifier demo mode (70% compliant)")
        print(f"[INFO] Real model will be at: {model_path}")

    def classify(self, cropped_bgr: np.ndarray) -> Dict[str, Any]:
        """Return simulated PPE classification for demo/testing."""
        if cropped_bgr is None or cropped_bgr.size == 0:
            return {
                "suit": None,
                "shield": None,
                "gloves": None,
                "confidences": {"suit": 0.0, "shield": 0.0, "gloves": 0.0},
                "all_compliant": None,
                "inference_time_ms": 0.0,
            }
        
        # Simulate: 70% of time all compliant, 30% randomly non-compliant
        import random
        compliant = random.random() < self.compliant_rate
        
        if compliant:
            suit_c, shield_c, gloves_c = 0.85, 0.82, 0.90
        else:
            # Randomly missing one piece
            missing = random.choice(['suit', 'shield', 'gloves'])
            suit_c = 0.2 if missing == 'suit' else 0.85
            shield_c = 0.25 if missing == 'shield' else 0.82
            gloves_c = 0.15 if missing == 'gloves' else 0.90
        
        return {
            "suit": suit_c > 0.65,
            "shield": shield_c > 0.65,
            "gloves": gloves_c > 0.65,
            "confidences": {
                "suit": round(suit_c, 3),
                "shield": round(shield_c, 3),
                "gloves": round(gloves_c, 3),
            },
            "all_compliant": compliant,
            "inference_time_ms": random.uniform(2, 8),
        }


class PPEClassifier:
    """
    TFLite-based PPE classifier for suit, shield, and gloves detection.
    Supports real inference, placeholder mode (A8), and tflite_runtime fallback.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.65, use_placeholder: bool = False) -> None:
        """
        Initialize PPE classifier with TFLite model or placeholder.

        Args:
            model_path: Path to .tflite model file
            conf_threshold: Confidence threshold for boolean classification
            use_placeholder: FIX A8 - Force placeholder mode for demo testing
        """
        from pathlib import Path
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._mock = False
        self._placeholder = use_placeholder
        
        # FIX A8: Check if model file exists
        if not Path(model_path).exists():
            if use_placeholder:
                print(f"[A8-PLACEHOLDER] Model not found, using placeholder classifier")
                print(f"[INFO] Real model expected at: {model_path}")
                self._placeholder = True
                self._classifier = PPEClassifierMock(model_path, conf_threshold)
                return
            else:
                print(f"[WARNING] Model not found at {model_path}")
                print(f"[INFO] Options:")
                print(f"[INFO]   1. Run Roboflow training (see roboflow_upload_guide.txt)")
                print(f"[INFO]   2. Use placeholder mode: PPEClassifier(..., use_placeholder=True)")
                print(f"[WARNING] Falling back to placeholder mode")
                self._placeholder = True
                self._classifier = PPEClassifierMock(model_path, conf_threshold)
                return

        if not TFLITE_AVAILABLE:
            print("[WARNING] tflite_runtime not installed. Using placeholder classifier.")
            print("[INFO] Install: pip install tflite-runtime")
            self._placeholder = True
            self._classifier = PPEClassifierMock(model_path, conf_threshold)
            return

        self._mock = False
        self._placeholder = False

        # Load model (FIX A8+A9: Better error handling)
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            print(f"[INFO] Loaded TFLite model: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load TFLite model: {e}")
            print(f"[INFO] Falling back to placeholder mode")
            self._placeholder = True
            self._classifier = PPEClassifierMock(model_path, conf_threshold)
            return

        # Cache tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input tensor info
        self.input_shape = self.input_details[0]["shape"]  # (1, 224, 224, 3)
        self.input_dtype = self.input_details[0]["dtype"]

        # Model expects 224x224 input (standard for MobileNetV3)
        self.input_size = 224

        print(f"[INFO] Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"[INFO] Output details: {len(self.output_details)} tensor(s)")

    def classify(self, cropped_bgr: np.ndarray) -> Dict[str, Any]:
        """
        FIX A8+A9: Classify PPE in cropped BGR image.
        Handles real model, placeholder mode, and multi-output formats.

        Args:
            cropped_bgr: BGR image of person/torso (any size, will be resized)

        Returns:
            Dict with keys:
            - suit: bool or None
            - shield: bool or None
            - gloves: bool or None
            - confidences: dict with float values
            - all_compliant: bool (all True) or None
            - inference_time_ms: float
        """
        # FIX A8: Use placeholder if in placeholder mode
        if self._placeholder or self._mock:
            return self._classifier.classify(cropped_bgr)

        start_time = time.time()

        # Validate input
        if cropped_bgr is None or cropped_bgr.size == 0:
            return self._null_result("empty_crop", 0.0)

        h, w = cropped_bgr.shape[:2]
        if w < 30 or h < 50 or w * h < 1800:
            return self._null_result("too_small", 0.0)

        try:
            # Preprocess
            input_tensor = self._preprocess(cropped_bgr)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()

            # FIX A9: Handle both single-output and 3-output models
            if len(self.output_details) == 1:
                # Single output tensor with 3 values [suit, shield, gloves]
                output_data = self.interpreter.get_tensor(
                    self.output_details[0]["index"]
                )
                confidences = self._sigmoid(output_data[0])
                suit_conf = float(confidences[0])
                shield_conf = float(confidences[1])
                gloves_conf = float(confidences[2])
            else:
                # Three separate output tensors
                suit_conf = self._sigmoid(np.array([self.interpreter.get_tensor(
                    self.output_details[0]["index"])[0][0]]))[0]
                shield_conf = self._sigmoid(np.array([self.interpreter.get_tensor(
                    self.output_details[1]["index"])[0][0]]))[0]
                gloves_conf = self._sigmoid(np.array([self.interpreter.get_tensor(
                    self.output_details[2]["index"])[0][0]]))[0]

            # Apply face shield heuristic fallback
            if 0.35 <= shield_conf <= 0.65:
                shield_heuristic = self._detect_shield_strap(cropped_bgr)
                if shield_heuristic:
                    shield_conf = 0.9
                    print("[INFO] Shield heuristic detected strap")

            # Convert to boolean
            suit = suit_conf > self.conf_threshold
            shield = shield_conf > self.conf_threshold
            gloves = gloves_conf > self.conf_threshold

            inference_time_ms = (time.time() - start_time) * 1000.0

            return {
                "suit": suit,
                "shield": shield,
                "gloves": gloves,
                "confidences": {
                    "suit": round(suit_conf, 3),
                    "shield": round(shield_conf, 3),
                    "gloves": round(gloves_conf, 3),
                },
                "all_compliant": suit and shield and gloves,
                "inference_time_ms": round(inference_time_ms, 1),
            }

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return self._null_result("inference_error", (time.time() - start_time) * 1000)

    def _null_result(self, reason: str, inference_time_ms: float = 0.0) -> Dict[str, Any]:
        """FIX A8: Return null result with reason."""
        return {
            "suit": None,
            "shield": None,
            "gloves": None,
            "confidences": {"suit": 0.0, "shield": 0.0, "gloves": 0.0},
            "all_compliant": None,
            "inference_time_ms": round(inference_time_ms, 1),
            "null_reason": reason,
        }


    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR image for MobileNetV3 inference.

        Args:
            bgr_image: BGR image of any size

        Returns:
            Preprocessed tensor ready for inference: (1, 224, 224, 3)
        """
        # Resize to 224x224
        rgb_resized = cv2.resize(bgr_image, (self.input_size, self.input_size))

        # BGR → RGB
        rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0.0, 1.0]
        rgb_normalized = rgb.astype(np.float32) / 255.0

        # Expand dims to (1, 224, 224, 3)
        input_tensor = np.expand_dims(rgb_normalized, axis=0)

        return input_tensor

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation to logits (multi-label).

        Args:
            x: Logits array

        Returns:
            Sigmoid-activated probabilities
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _detect_shield_strap(bgr_image: np.ndarray) -> bool:
        """
        Detect face shield strap using Hough line detection.
        Looks for horizontal lines in top 25% of image.

        Args:
            bgr_image: BGR image

        Returns:
            True if shield strap detected, False otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            # Focus on top 25% (face area where strap would be)
            top_height = max(1, int(gray.shape[0] * 0.25))
            gray_top = gray[:top_height, :]

            # Apply edge detection
            edges = cv2.Canny(gray_top, 50, 150)

            # Detect horizontal lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10,
            )

            if lines is not None:
                # Count horizontal lines (small angle variance)
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1))
                    # Horizontal = angle near 0 or 180 degrees
                    if angle < 0.3 or angle > np.pi - 0.3:
                        horizontal_lines += 1

                return horizontal_lines >= 2

        except Exception as e:
            print(f"[WARNING] Shield strap detection failed: {e}")
            pass

        return False


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ppe_classifier.py <image_path> [model_path]")
        print("Example: python ppe_classifier.py person.jpg ./model.tflite")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "./model.tflite"

    # Load image
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] Could not load image: {image_path}")
            sys.exit(1)
        print(f"[INFO] Loaded image: {image_path}, shape: {image_bgr.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        sys.exit(1)

    # Initialize classifier
    try:
        classifier = PPEClassifier(model_path=model_path, conf_threshold=0.65)
    except Exception as e:
        print(f"[ERROR] Failed to initialize classifier: {e}")
        sys.exit(1)

    # Classify
    result = classifier.classify(image_bgr)
    print("\n[RESULT]")
    print(f"Suit: {result['suit']} (conf: {result['confidences']['suit']:.3f})")
    print(f"Shield: {result['shield']} (conf: {result['confidences']['shield']:.3f})")
    print(f"Gloves: {result['gloves']} (conf: {result['confidences']['gloves']:.3f})")
    print(f"All Compliant: {result['all_compliant']}")
    print(f"Inference Time: {result['inference_time_ms']:.2f} ms")

    # Annotate and display
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    label_text = f"Suit:{result['suit']} Shield:{result['shield']} Gloves:{result['gloves']}"
    cv2.putText(
        annotated,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if result["all_compliant"] else (0, 0, 255),
        2,
    )

    cv2.imshow("PPE Classification", annotated)
    print("\n[INFO] Displaying image. Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
