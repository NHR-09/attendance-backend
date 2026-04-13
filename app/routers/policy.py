"""
Policy configuration router — get/update attendance policies.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.models import Employee, PolicyConfig
from app.schemas.schemas import PolicyConfigOut, PolicyConfigUpdate
from app.utils.auth import require_role, get_current_user
from app.services.audit_service import log_event

router = APIRouter(prefix="/api/policy", tags=["Policy"])

# Default policy values
DEFAULT_POLICIES = {
    "cutoff_time": {"value": "18:00", "description": "Auto not-checked-out cutoff time (HH:MM)"},
    "confidence_threshold": {"value": "0.6", "description": "Minimum face match confidence (0-1)"},
    "geofence_lat": {"value": "28.6139", "description": "Office latitude for geofence center"},
    "geofence_lng": {"value": "77.2090", "description": "Office longitude for geofence center"},
    "geofence_radius": {"value": "200", "description": "Allowed radius in meters from office"},
    "location_fallback_enabled": {"value": "true", "description": "Allow location-based fallback attendance"},
}


@router.get("/", response_model=list[PolicyConfigOut])
async def list_policies(
    db: AsyncSession = Depends(get_db),
    _user: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(select(PolicyConfig).order_by(PolicyConfig.key))
    policies = result.scalars().all()

    # If empty, seed defaults
    if not policies:
        for key, data in DEFAULT_POLICIES.items():
            p = PolicyConfig(key=key, value=data["value"], description=data["description"])
            db.add(p)
        await db.flush()
        result = await db.execute(select(PolicyConfig).order_by(PolicyConfig.key))
        policies = result.scalars().all()

    return policies


@router.put("/{key}", response_model=PolicyConfigOut)
async def update_policy(
    key: str,
    payload: PolicyConfigUpdate,
    db: AsyncSession = Depends(get_db),
    manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(select(PolicyConfig).where(PolicyConfig.key == key))
    policy = result.scalar_one_or_none()
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy '{key}' not found")

    old_value = policy.value
    policy.value = payload.value
    if payload.description:
        policy.description = payload.description
    policy.updated_by = manager.id

    await log_event(
        db, manager.id, "update_policy", "policy_config", policy.id,
        old_value, payload.value,
    )

    return policy
