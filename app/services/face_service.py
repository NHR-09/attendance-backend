"""
Face verification service — STUBBED for development.
Replace the compare() function with InsightFace/FaceNet in production.
"""
import random
import numpy as np
from typing import Optional


def generate_face_encoding(image_bytes: bytes) -> bytes:
    """Generate a fake face encoding from image bytes.
    In production: use InsightFace to extract a 512-d embedding vector.
    """
    # Stub: generate a deterministic-ish 128-d random vector
    rng = np.random.RandomState(hash(image_bytes[:64]) % (2**31))
    encoding = rng.rand(128).astype(np.float32)
    return encoding.tobytes()


def compare_faces(
    stored_encoding: Optional[bytes],
    live_encoding: bytes,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Compare a stored face encoding against a live capture encoding.
    Returns (matched: bool, confidence: float 0-100).

    STUB: returns a high-confidence match most of the time.
    """
    if stored_encoding is None:
        return False, 0.0

    # Stub: simulate a realistic confidence score
    confidence = round(random.uniform(78.0, 99.5), 2)
    matched = confidence >= (threshold * 100)
    return matched, confidence


def check_liveness(image_bytes: bytes) -> bool:
    """Basic liveness/anti-spoofing check.
    STUB: always returns True. Replace with real anti-spoof in production.
    """
    return True
