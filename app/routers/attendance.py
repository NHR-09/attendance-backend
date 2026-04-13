"""
Attendance router — check-in, check-out, status, history.
"""
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import os, uuid

from app.database import get_db
from app.config import get_settings
from app.models.models import (
    Employee, AttendanceLog, DeviceBinding, PhotoEvidence,
    PolicyConfig, AttendanceMethod, AttendanceStatus,
)
from app.schemas.schemas import CheckInRequest, CheckOutRequest, AttendanceLogOut
from app.utils.auth import get_current_user
from app.services.audit_service import log_event
from app.services.face_service import generate_face_encoding, compare_faces, check_liveness
from app.services.location_service import validate_location

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])
settings = get_settings()


async def _get_policy(db: AsyncSession, key: str, default: str) -> str:
    r = await db.execute(select(PolicyConfig).where(PolicyConfig.key == key))
    p = r.scalar_one_or_none()
    return p.value if p else default


async def _get_geofence(db: AsyncSession):
    lat = float(await _get_policy(db, "geofence_lat", "28.6139"))
    lng = float(await _get_policy(db, "geofence_lng", "77.2090"))
    radius = float(await _get_policy(db, "geofence_radius", "200"))
    return lat, lng, radius


async def _verify_device(db: AsyncSession, employee: Employee, device_id: str):
    result = await db.execute(
        select(DeviceBinding).where(
            and_(
                DeviceBinding.employee_id == employee.id,
                DeviceBinding.device_id == device_id,
                DeviceBinding.is_active == True,
            )
        )
    )
    binding = result.scalar_one_or_none()
    if not binding:
        raise HTTPException(
            status_code=403,
            detail="Device not registered. Please bind this device first or contact your manager.",
        )
    return binding


@router.post("/check-in", response_model=AttendanceLogOut)
async def check_in(
    device_id: str = Form(...),
    method: str = Form("face"),
    confidence_score: float = Form(None),
    location_lat: float = Form(None),
    location_lng: float = Form(None),
    photo: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    # 1. Verify device binding
    await _verify_device(db, current_user, device_id)

    # 2. Check if already checked in today
    today = date.today()
    existing = await db.execute(
        select(AttendanceLog).where(
            and_(
                AttendanceLog.employee_id == current_user.id,
                AttendanceLog.date == today,
            )
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Already checked in today")

    # 3. Read policy values
    threshold = float(await _get_policy(db, "confidence_threshold", "0.6")) * 100
    geo_lat, geo_lng, geo_radius = await _get_geofence(db)

    final_confidence = confidence_score
    att_method = AttendanceMethod.FACE if method == "face" else AttendanceMethod.LOCATION

    if method == "face" and photo:
        image_bytes = await photo.read()

        # Bug #10: liveness check
        if not check_liveness(image_bytes):
            raise HTTPException(status_code=400, detail="Liveness check failed. Please use a live camera.")

        live_encoding = generate_face_encoding(image_bytes)
        matched, conf = compare_faces(current_user.face_encoding, live_encoding, threshold / 100)
        final_confidence = conf

        if not matched:
            # Bug #1: face fail must validate location, not silently pass
            if location_lat is None or location_lng is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Face verification failed ({conf:.1f}% confidence). Enable location fallback to proceed.",
                )
            within, dist = validate_location(location_lat, location_lng, geo_lat, geo_lng, geo_radius)
            if not within:
                raise HTTPException(
                    status_code=400,
                    detail=f"Face failed and outside geofence ({dist:.0f}m away).",
                )
            att_method = AttendanceMethod.LOCATION

    elif method == "location":
        if location_lat is None or location_lng is None:
            raise HTTPException(status_code=400, detail="Location coordinates required for fallback")
        # Bug #2: use geofence values from DB
        within, dist = validate_location(location_lat, location_lng, geo_lat, geo_lng, geo_radius)
        if not within:
            raise HTTPException(
                status_code=400,
                detail=f"Outside allowed area. Distance: {dist:.0f}m",
            )

    # 4. Save attendance
    log = AttendanceLog(
        employee_id=current_user.id,
        date=today,
        check_in_time=datetime.utcnow(),
        confidence_score=final_confidence,
        method=att_method,
        status=AttendanceStatus.PRESENT,
        device_id=device_id,
        location_lat=location_lat,
        location_lng=location_lng,
    )
    db.add(log)
    await db.flush()
    await db.refresh(log)

    # 5. Save photo evidence
    if photo:
        await photo.seek(0)
        image_bytes = await photo.read()
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        evidence = PhotoEvidence(
            attendance_log_id=log.id,
            image_path=filepath,
            confidence_score=final_confidence,
            is_low_confidence=(final_confidence or 0) < threshold,
        )
        db.add(evidence)

    await log_event(db, current_user.id, "check_in", "attendance_log", log.id)

    return AttendanceLogOut(
        id=log.id,
        employee_id=log.employee_id,
        employee_name=current_user.name,
        employee_code=current_user.employee_id,
        date=log.date,
        check_in_time=log.check_in_time,
        check_out_time=log.check_out_time,
        confidence_score=log.confidence_score,
        method=log.method.value,
        status=log.status.value,
        device_id=log.device_id,
        location_lat=log.location_lat,
        location_lng=log.location_lng,
        notes=log.notes,
        created_at=log.created_at,
    )


@router.post("/check-out", response_model=AttendanceLogOut)
async def check_out(
    device_id: str = Form(...),
    method: str = Form("face"),
    confidence_score: float = Form(None),
    photo: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    await _verify_device(db, current_user, device_id)

    today = date.today()
    result = await db.execute(
        select(AttendanceLog).where(
            and_(
                AttendanceLog.employee_id == current_user.id,
                AttendanceLog.date == today,
            )
        )
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=400, detail="No check-in found for today")
    if log.check_out_time:
        raise HTTPException(status_code=400, detail="Already checked out today")

    # Bug #3: read threshold from DB
    threshold = float(await _get_policy(db, "confidence_threshold", "0.6")) * 100

    final_confidence = confidence_score
    if method == "face" and photo:
        image_bytes = await photo.read()

        # Bug #10: liveness check on checkout too
        if not check_liveness(image_bytes):
            raise HTTPException(status_code=400, detail="Liveness check failed.")

        live_encoding = generate_face_encoding(image_bytes)
        matched, conf = compare_faces(current_user.face_encoding, live_encoding, threshold / 100)
        final_confidence = conf

        # Bug #4: reject failed face on checkout
        if not matched:
            raise HTTPException(
                status_code=400,
                detail=f"Face verification failed ({conf:.1f}%). Please try again.",
            )

    log.check_out_time = datetime.utcnow()
    log.confidence_score = final_confidence or log.confidence_score

    if photo:
        await photo.seek(0)
        image_bytes = await photo.read()
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_out.jpg"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        evidence = PhotoEvidence(
            attendance_log_id=log.id,
            image_path=filepath,
            confidence_score=final_confidence,
            is_low_confidence=(final_confidence or 0) < threshold,
        )
        db.add(evidence)

    await log_event(db, current_user.id, "check_out", "attendance_log", log.id)

    return AttendanceLogOut(
        id=log.id,
        employee_id=log.employee_id,
        employee_name=current_user.name,
        employee_code=current_user.employee_id,
        date=log.date,
        check_in_time=log.check_in_time,
        check_out_time=log.check_out_time,
        confidence_score=log.confidence_score,
        method=log.method.value,
        status=log.status.value,
        device_id=log.device_id,
        location_lat=log.location_lat,
        location_lng=log.location_lng,
        notes=log.notes,
        created_at=log.created_at,
    )


@router.get("/status/today", response_model=AttendanceLogOut | None)
async def get_today_status(
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    today = date.today()
    result = await db.execute(
        select(AttendanceLog).where(
            and_(
                AttendanceLog.employee_id == current_user.id,
                AttendanceLog.date == today,
            )
        )
    )
    log = result.scalar_one_or_none()
    if not log:
        return None

    return AttendanceLogOut(
        id=log.id,
        employee_id=log.employee_id,
        employee_name=current_user.name,
        employee_code=current_user.employee_id,
        date=log.date,
        check_in_time=log.check_in_time,
        check_out_time=log.check_out_time,
        confidence_score=log.confidence_score,
        method=log.method.value,
        status=log.status.value,
        device_id=log.device_id,
        location_lat=log.location_lat,
        location_lng=log.location_lng,
        notes=log.notes,
        created_at=log.created_at,
    )


@router.get("/history", response_model=list[AttendanceLogOut])
async def get_history(
    limit: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    result = await db.execute(
        select(AttendanceLog)
        .where(AttendanceLog.employee_id == current_user.id)
        .order_by(AttendanceLog.date.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        AttendanceLogOut(
            id=l.id,
            employee_id=l.employee_id,
            employee_name=current_user.name,
            employee_code=current_user.employee_id,
            date=l.date,
            check_in_time=l.check_in_time,
            check_out_time=l.check_out_time,
            confidence_score=l.confidence_score,
            method=l.method.value,
            status=l.status.value,
            device_id=l.device_id,
            location_lat=l.location_lat,
            location_lng=l.location_lng,
            notes=l.notes,
            created_at=l.created_at,
        )
        for l in logs
    ]
