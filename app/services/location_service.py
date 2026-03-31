"""
Location validation service — Geofence check using haversine distance.
"""
import math
from typing import Optional


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_location(
    employee_lat: float,
    employee_lng: float,
    office_lat: float = 28.6139,   # Default: New Delhi
    office_lng: float = 77.2090,
    allowed_radius_m: float = 200.0,
) -> tuple[bool, float]:
    """Check if the employee is within the allowed geofence.
    Returns (within_fence: bool, distance_m: float).
    """
    distance = haversine_distance(employee_lat, employee_lng, office_lat, office_lng)
    return distance <= allowed_radius_m, round(distance, 2)
