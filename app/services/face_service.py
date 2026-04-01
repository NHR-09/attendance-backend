"""
Face verification service — OpenCV-powered (no TensorFlow required).
Uses OpenCV DNN face detector + normalized pixel embeddings for matching.
Lightweight enough for Render free tier.
"""
import numpy as np
from typing import Optional
import cv2


# ── Face Detection ──────────────────────────────────────────────────
# Use OpenCV's Haar cascade (bundled with opencv-python-headless)
_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _detect_and_crop_face(img: np.ndarray, target_size: int = 160) -> np.ndarray:
    """Detect a face and return a cropped, resized, grayscale face region.
    Raises ValueError if no face detected.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        raise ValueError("No face detected. Please ensure your face is clearly visible.")

    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Add padding (20%)
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    face_roi = gray[y1:y2, x1:x2]
    # Resize to standard size and normalize
    face_resized = cv2.resize(face_roi, (target_size, target_size))
    return face_resized


def _face_to_embedding(face_img: np.ndarray) -> np.ndarray:
    """Convert a face image to a normalized embedding vector.
    Uses histogram of oriented gradients (HOG) features + pixel stats.
    """
    # 1. Histogram equalization for lighting normalization
    equalized = cv2.equalizeHist(face_img)

    # 2. Compute HOG-like features using Sobel gradients
    gx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    # 3. Divide into 8x8 grid cells and compute gradient histograms
    cell_size = face_img.shape[0] // 8
    features = []
    for i in range(8):
        for j in range(8):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            # 9-bin histogram of gradient orientations
            hist, _ = np.histogram(cell_ang, bins=9, range=(-np.pi, np.pi), weights=cell_mag)
            features.extend(hist)

    # 4. Add pixel intensity statistics per cell
    for i in range(8):
        for j in range(8):
            cell = equalized[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size].astype(np.float64)
            features.append(np.mean(cell))
            features.append(np.std(cell))

    embedding = np.array(features, dtype=np.float32)
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ── Public API ──────────────────────────────────────────────────────

def generate_face_encoding(image_bytes: bytes) -> bytes:
    """Extract a face embedding from image bytes.
    Returns the embedding as raw bytes.
    Raises ValueError if no face is detected.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    face_img = _detect_and_crop_face(img)
    embedding = _face_to_embedding(face_img)
    return embedding.tobytes()


def compare_faces(
    stored_encoding: Optional[bytes],
    live_encoding: bytes,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Compare stored face embedding against live capture using cosine similarity.
    Returns (matched: bool, confidence: float 0-100).
    """
    if stored_encoding is None:
        return False, 0.0

    try:
        stored = np.frombuffer(stored_encoding, dtype=np.float32)
        live = np.frombuffer(live_encoding, dtype=np.float32)

        if stored.shape != live.shape:
            return False, 0.0

        # Cosine similarity (embeddings are already L2-normalized)
        cosine_sim = float(np.dot(stored, live))

        # Map cosine similarity to confidence percentage
        # Typical same-person scores: 0.7-0.95
        # Typical different-person scores: 0.3-0.6
        confidence = max(0.0, min(100.0, cosine_sim * 100))
        confidence = round(confidence, 2)

        matched = confidence >= (threshold * 100)
        return matched, confidence

    except Exception:
        return False, 0.0


def check_liveness(image_bytes: bytes) -> bool:
    """Basic anti-spoofing checks:
    1. Exactly one face detected (reject zero or multiple)
    2. Image not too blurry (reject screen photos)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) != 1:
            return False

        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False

        return True

    except Exception:
        return False
