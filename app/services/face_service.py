"""
Face verification service — DeepFace-powered face recognition.
Uses Facenet512 model for embedding extraction + cosine similarity for matching.
"""
import io
import numpy as np
from typing import Optional
from PIL import Image
import cv2

# Lazy-load DeepFace to avoid slow startup
_deepface = None

def _get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def generate_face_encoding(image_bytes: bytes) -> bytes:
    """Extract a face embedding from image bytes using DeepFace (Facenet512).
    Returns the embedding as raw bytes (512 floats = 2048 bytes).
    Raises ValueError if no face is detected.
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        DeepFace = _get_deepface()
        # Extract embedding using Facenet512 (512-d vector)
        embeddings = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv",
        )

        if not embeddings or len(embeddings) == 0:
            raise ValueError("No face detected in image")

        # Return the first face's embedding as bytes
        embedding = np.array(embeddings[0]["embedding"], dtype=np.float32)
        return embedding.tobytes()

    except Exception as e:
        error_msg = str(e)
        if "Face could not be detected" in error_msg or "No face" in error_msg:
            raise ValueError("No face detected. Please ensure your face is clearly visible.")
        raise ValueError(f"Face encoding failed: {error_msg}")


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
        # Convert bytes back to numpy arrays
        stored = np.frombuffer(stored_encoding, dtype=np.float32)
        live = np.frombuffer(live_encoding, dtype=np.float32)

        # Ensure same dimensions
        if stored.shape != live.shape:
            return False, 0.0

        # Cosine similarity
        dot_product = np.dot(stored, live)
        norm_stored = np.linalg.norm(stored)
        norm_live = np.linalg.norm(live)

        if norm_stored == 0 or norm_live == 0:
            return False, 0.0

        cosine_sim = dot_product / (norm_stored * norm_live)

        # Convert cosine similarity (0-1) to confidence percentage (0-100)
        # Facenet512 typically gives cosine similarity > 0.6 for same person
        confidence = round(float(cosine_sim) * 100, 2)
        confidence = max(0.0, min(100.0, confidence))

        matched = confidence >= (threshold * 100)
        return matched, confidence

    except Exception:
        return False, 0.0


def check_liveness(image_bytes: bytes) -> bool:
    """Basic anti-spoofing checks:
    1. Detect exactly one face (reject multiple faces)
    2. Check image isn't too blurry (reject screen photos)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Face count check — reject if 0 or >1 faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) != 1:
            return False

        # 2. Blur detection — Laplacian variance (low = blurry)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # very blurry, likely a screen photo
            return False

        return True

    except Exception:
        return False
