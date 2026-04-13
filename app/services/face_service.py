"""
Face verification service — OpenCV DNN-powered.
Uses YuNet face detector + SFace recognizer for identity-aware embeddings.
Much more discriminative than basic HOG — can actually distinguish people.
"""
import os
import urllib.request
import numpy as np
import cv2
from typing import Optional

# ── Model URLs (from OpenCV Zoo) ────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_DETECTOR_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_RECOGNIZER_PATH = os.path.join(_MODEL_DIR, "face_recognition_sface_2021dec.onnx")

_detector = None
_recognizer = None


def _ensure_models():
    """Download DNN models if not cached locally."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for url, path in [(_DETECTOR_URL, _DETECTOR_PATH), (_RECOGNIZER_URL, _RECOGNIZER_PATH)]:
        if not os.path.exists(path):
            print(f"[face_service] Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"[face_service] Saved to {path}")


def _get_detector(width: int, height: int):
    """Get or create YuNet face detector."""
    global _detector
    _ensure_models()
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(_DETECTOR_PATH, "", (width, height))
    else:
        _detector.setInputSize((width, height))
    return _detector


def _get_recognizer():
    """Get or create SFace recognizer."""
    global _recognizer
    _ensure_models()
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(_RECOGNIZER_PATH, "")
    return _recognizer


def _detect_face(img: np.ndarray) -> np.ndarray:
    """Detect faces and return the largest face bounding info.
    Returns the raw face detection array needed by SFace.
    Raises ValueError if no face detected.
    """
    h, w = img.shape[:2]
    detector = _get_detector(w, h)
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected. Please ensure your face is clearly visible and well-lit.")

    # Take the largest face (by area = w*h)
    areas = faces[:, 2] * faces[:, 3]
    largest_idx = np.argmax(areas)
    return faces[largest_idx]


def _align_and_embed(img: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Align face and extract 128-d embedding using SFace."""
    recognizer = _get_recognizer()
    aligned = recognizer.alignCrop(img, face)
    embedding = recognizer.feature(aligned)
    return embedding.flatten()


# ── Public API ──────────────────────────────────────────────────────

def generate_face_encoding(image_bytes: bytes) -> bytes:
    """Extract a 128-d face embedding from image bytes.
    Returns the embedding as raw bytes (512 bytes = 128 x float32).
    Raises ValueError if no face is detected.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    face = _detect_face(img)
    embedding = _align_and_embed(img, face)

    # L2-normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32).tobytes()


def compare_faces(
    stored_encoding: Optional[bytes],
    live_encoding: bytes,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Compare stored face embedding against live capture using cosine similarity.
    Returns (matched: bool, confidence: float 0-100).

    SFace cosine similarity scores:
    - Same person: typically 0.5 - 0.8+
    - Different person: typically 0.0 - 0.3
    """
    if stored_encoding is None:
        return False, 0.0

    try:
        stored = np.frombuffer(stored_encoding, dtype=np.float32)
        live = np.frombuffer(live_encoding, dtype=np.float32)

        if stored.shape != live.shape:
            return False, 0.0

        # Cosine similarity (embeddings are L2-normalized)
        cosine_sim = float(np.dot(stored, live))

        # Map to confidence percentage
        # SFace cosine scores range ~0.0 to ~0.8
        # Scale so that 0.4 similarity → ~60%, 0.6 → ~85%, 0.8 → ~100%
        confidence = max(0.0, min(100.0, cosine_sim * 120))
        confidence = round(confidence, 2)

        matched = confidence >= (threshold * 100)
        return matched, confidence

    except Exception:
        return False, 0.0


def check_liveness(image_bytes: bytes) -> bool:
    """Anti-spoofing checks:
    1. Exactly one face detected (reject zero or multiple)
    2. Image not too blurry (reject screen photos)
    3. Face is large enough in frame (reject distant/tiny faces)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]
        detector = _get_detector(w, h)
        _, faces = detector.detect(img)

        if faces is None or len(faces) != 1:
            return False

        face = faces[0]
        face_w, face_h = face[2], face[3]

        # Face must be at least 10% of image width (not too far away)
        if face_w < w * 0.1 or face_h < h * 0.1:
            return False

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 15:
            return False

        return True

    except Exception:
        return False
