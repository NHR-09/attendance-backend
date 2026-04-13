"""
Face verification service — OpenCV DNN-powered.
Uses YuNet face detector + SFace recognizer for identity-aware embeddings.
Falls back to Haar cascade if DNN models aren't available.
"""
import os
import urllib.request
import numpy as np
import cv2
from typing import Optional

# ── Model paths ─────────────────────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_DETECTOR_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_RECOGNIZER_PATH = os.path.join(_MODEL_DIR, "face_recognition_sface_2021dec.onnx")

_detector = None
_recognizer = None
_haar_cascade = None
_use_dnn = None  # None = not yet determined, True/False after first attempt


def _download_file(url: str, path: str):
    """Download a file following redirects."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[face_service] Downloading {os.path.basename(path)}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
        # Verify it's not an HTML error page
        if len(data) < 10000 or data[:4] == b'<htm' or data[:4] == b'<!DO':
            raise ValueError(f"Downloaded file appears to be HTML, not ONNX model ({len(data)} bytes)")
        with open(path, 'wb') as f:
            f.write(data)
    print(f"[face_service] Saved {os.path.basename(path)} ({len(data)} bytes)")


def _ensure_models() -> bool:
    """Download DNN models if not cached. Returns True if models are available."""
    global _use_dnn
    if _use_dnn is not None:
        return _use_dnn

    try:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        for url, path in [(_DETECTOR_URL, _DETECTOR_PATH), (_RECOGNIZER_URL, _RECOGNIZER_PATH)]:
            if not os.path.exists(path) or os.path.getsize(path) < 10000:
                _download_file(url, path)
        # Verify models can be loaded
        test_det = cv2.FaceDetectorYN.create(_DETECTOR_PATH, "", (320, 320))
        test_rec = cv2.FaceRecognizerSF.create(_RECOGNIZER_PATH, "")
        _use_dnn = True
        print("[face_service] DNN models loaded successfully")
    except Exception as e:
        print(f"[face_service] DNN models unavailable, using Haar fallback: {e}")
        _use_dnn = False

    return _use_dnn


def _get_haar_cascade():
    """Fallback face detector."""
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _haar_cascade


def _get_detector(width: int, height: int):
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            _DETECTOR_PATH, "", (width, height), 0.5, 0.3, 5000
        )
    else:
        _detector.setInputSize((width, height))
    return _detector


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(_RECOGNIZER_PATH, "")
    return _recognizer


# ── DNN-based detection + embedding ─────────────────────────────────

def _detect_face_dnn(img: np.ndarray) -> np.ndarray:
    """Detect face using YuNet DNN. Returns face detection array."""
    h, w = img.shape[:2]
    detector = _get_detector(w, h)
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        raise ValueError("No face detected. Please ensure your face is clearly visible and well-lit.")
    areas = faces[:, 2] * faces[:, 3]
    return faces[np.argmax(areas)]


def _embed_face_dnn(img: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Extract 128-d SFace embedding."""
    recognizer = _get_recognizer()
    aligned = recognizer.alignCrop(img, face)
    embedding = recognizer.feature(aligned).flatten()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


# ── Haar fallback detection + embedding ──────────────────────────────

def _detect_and_crop_haar(img: np.ndarray, target_size: int = 160) -> np.ndarray:
    """Detect face with Haar cascade, return cropped grayscale face."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _get_haar_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No face detected. Please ensure your face is clearly visible and well-lit.")
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    pad = int(0.2 * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    return cv2.resize(gray[y1:y2, x1:x2], (target_size, target_size))


def _embed_face_haar(face_img: np.ndarray) -> np.ndarray:
    """HOG-based embedding (less accurate fallback)."""
    equalized = cv2.equalizeHist(face_img)
    gx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    cell_size = face_img.shape[0] // 8
    features = []
    for i in range(8):
        for j in range(8):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(cell_ang, bins=9, range=(-np.pi, np.pi), weights=cell_mag)
            features.extend(hist)
    for i in range(8):
        for j in range(8):
            cell = equalized[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size].astype(np.float64)
            features.append(np.mean(cell))
            features.append(np.std(cell))

    embedding = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ── Public API ──────────────────────────────────────────────────────

def generate_face_encoding(image_bytes: bytes) -> bytes:
    """Extract face embedding from image bytes. Returns raw bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    if _ensure_models():
        face = _detect_face_dnn(img)
        embedding = _embed_face_dnn(img, face)
    else:
        face_crop = _detect_and_crop_haar(img)
        embedding = _embed_face_haar(face_crop)

    return embedding.tobytes()


def compare_faces(
    stored_encoding: Optional[bytes],
    live_encoding: bytes,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Compare face embeddings using cosine similarity.
    Returns (matched: bool, confidence: float 0-100).
    """
    if stored_encoding is None:
        return False, 0.0

    try:
        stored = np.frombuffer(stored_encoding, dtype=np.float32)
        live = np.frombuffer(live_encoding, dtype=np.float32)

        if stored.shape != live.shape:
            return False, 0.0

        cosine_sim = float(np.dot(stored, live))

        # Scale confidence based on embedding type
        if stored.shape[0] == 128:
            # SFace: scores range 0.0-0.8 for same person
            confidence = max(0.0, min(100.0, cosine_sim * 120))
        else:
            # Haar/HOG fallback: scores range 0.5-0.95
            confidence = max(0.0, min(100.0, cosine_sim * 100))

        confidence = round(confidence, 2)
        matched = confidence >= (threshold * 100)
        return matched, confidence

    except Exception:
        return False, 0.0


def check_liveness(image_bytes: bytes) -> bool:
    """Anti-spoofing: single face, not too blurry, reasonable size."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]

        if _ensure_models():
            detector = _get_detector(w, h)
            _, faces = detector.detect(img)
            if faces is None or len(faces) != 1:
                return False
            face_w, face_h = faces[0][2], faces[0][3]
            if face_w < w * 0.08 or face_h < h * 0.08:
                return False
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = _get_haar_cascade()
            faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
            if len(faces) != 1:
                return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 15:
            return False

        return True

    except Exception:
        return False
