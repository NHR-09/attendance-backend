"""
Manager router — attendance review, override, export, dashboard summary.
"""
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.database import get_db
from app.models.models import (
    Employee, AttendanceLog, PhotoEvidence, AuditEvent,
    AttendanceStatus,
)
from app.schemas.schemas import (
    AttendanceLogOut, AttendanceUpdate, PhotoEvidenceOut,
    AuditEventOut, DashboardSummary, ManualAttendanceCreate,
)
from app.utils.auth import require_role
from app.services.audit_service import log_event
from app.services.excel_service import generate_attendance_excel

router = APIRouter(prefix="/api/manager", tags=["Manager"])


# ── Dashboard Summary ──────────────────────────────────────────────
@router.get("/dashboard", response_model=DashboardSummary)
async def dashboard_summary(
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    today = date.today()

    total = await db.execute(select(func.count(Employee.id)).where(Employee.is_active == True))
    total_employees = total.scalar() or 0

    present = await db.execute(
        select(func.count(AttendanceLog.id)).where(
            and_(AttendanceLog.date == today, AttendanceLog.status == AttendanceStatus.PRESENT)
        )
    )
    present_today = present.scalar() or 0

    not_checked = await db.execute(
        select(func.count(AttendanceLog.id)).where(
            and_(AttendanceLog.date == today, AttendanceLog.status == AttendanceStatus.NOT_CHECKED_OUT)
        )
    )
    not_checked_out = not_checked.scalar() or 0

    exceptions = await db.execute(
        select(func.count(AttendanceLog.id)).where(
            and_(AttendanceLog.date == today, AttendanceLog.status == AttendanceStatus.EXCEPTION)
        )
    )
    exception_count = exceptions.scalar() or 0

    low_conf = await db.execute(
        select(func.count(PhotoEvidence.id))
        .join(AttendanceLog, PhotoEvidence.attendance_log_id == AttendanceLog.id)
        .where(and_(AttendanceLog.date == today, PhotoEvidence.is_low_confidence == True))
    )
    low_confidence_count = low_conf.scalar() or 0

    absent = await db.execute(
        select(func.count(AttendanceLog.id)).where(
            and_(AttendanceLog.date == today, AttendanceLog.status == AttendanceStatus.ABSENT)
        )
    )
    absent_today = absent.scalar() or 0

    return DashboardSummary(
        total_employees=total_employees,
        present_today=present_today,
        absent_today=max(absent_today, 0),
        not_checked_out=not_checked_out,
        exceptions=exception_count,
        low_confidence_count=low_confidence_count,
    )


# ── All Attendance Records ──────────────────────────────────────────
@router.get("/attendance", response_model=list[AttendanceLogOut])
async def list_attendance(
    target_date: date = Query(default=None),
    employee_id: int = Query(default=None),
    status_filter: str = Query(default=None),
    limit: int = Query(default=100),
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    query = select(AttendanceLog).order_by(AttendanceLog.date.desc(), AttendanceLog.check_in_time.desc())

    if target_date:
        query = query.where(AttendanceLog.date == target_date)
    if employee_id:
        query = query.where(AttendanceLog.employee_id == employee_id)
    if status_filter:
        try:
            query = query.where(AttendanceLog.status == AttendanceStatus(status_filter))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")

    query = query.limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()

    output = []
    emp_ids = list({l.employee_id for l in logs})
    emp_map = {}
    if emp_ids:
        emp_result = await db.execute(select(Employee).where(Employee.id.in_(emp_ids)))
        emp_map = {e.id: e for e in emp_result.scalars().all()}
    for l in logs:
        emp = emp_map.get(l.employee_id)
        output.append(AttendanceLogOut(
            id=l.id,
            employee_id=l.employee_id,
            employee_name=emp.name if emp else "",
            employee_code=emp.employee_id if emp else "",
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
        ))
    return output


# ── Edit Attendance ─────────────────────────────────────────────────
@router.put("/attendance/{log_id}", response_model=AttendanceLogOut)
async def edit_attendance(
    log_id: int,
    payload: AttendanceUpdate,
    db: AsyncSession = Depends(get_db),
    manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(select(AttendanceLog).where(AttendanceLog.id == log_id))
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    old_values = {}
    for field, val in payload.model_dump(exclude_unset=True, exclude={"reason"}).items():
        if val is not None:
            old_values[field] = str(getattr(log, field))
            setattr(log, field, val)

    await log_event(
        db, manager.id, "edit_attendance", "attendance_log", log.id,
        str(old_values),
        str(payload.model_dump(exclude_unset=True)),
        payload.reason,
    )

    emp_result = await db.execute(select(Employee).where(Employee.id == log.employee_id))
    emp = emp_result.scalar_one_or_none()

    return AttendanceLogOut(
        id=log.id,
        employee_id=log.employee_id,
        employee_name=emp.name if emp else "",
        employee_code=emp.employee_id if emp else "",
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


# ── Photo Evidence ──────────────────────────────────────────────────
@router.get("/photos/{log_id}", response_model=list[PhotoEvidenceOut])
async def get_photos(
    log_id: int,
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(
        select(PhotoEvidence).where(PhotoEvidence.attendance_log_id == log_id)
    )
    return result.scalars().all()


# ── Excel Export ────────────────────────────────────────────────────
@router.get("/export/excel")
async def export_excel(
    target_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    report_date = target_date or date.today()

    query = select(AttendanceLog).where(AttendanceLog.date == report_date)
    result = await db.execute(query)
    logs = result.scalars().all()

    # Batch-fetch employees to avoid N+1
    emp_ids = list({l.employee_id for l in logs})
    emp_map = {}
    if emp_ids:
        emp_result = await db.execute(select(Employee).where(Employee.id.in_(emp_ids)))
        emp_map = {e.id: e for e in emp_result.scalars().all()}

    records = []
    for l in logs:
        emp = emp_map.get(l.employee_id)
        records.append({
            "employee_name": emp.name if emp else "Unknown",
            "employee_id": emp.employee_id if emp else "",
            "confidence_score": l.confidence_score,
            "date": l.date,
            "check_in_time": l.check_in_time,
            "check_out_time": l.check_out_time,
            "method": l.method.value,
            "status": l.status.value,
            "notes": l.notes,
        })

    excel_file = generate_attendance_excel(records, report_date)

    return StreamingResponse(
        excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=attendance_{report_date}.xlsx"},
    )


# ── Audit Trail ─────────────────────────────────────────────────────
@router.get("/audit", response_model=list[AuditEventOut])
async def get_audit_trail(
    limit: int = Query(default=50),
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(
        select(AuditEvent).order_by(AuditEvent.timestamp.desc()).limit(limit)
    )
    return result.scalars().all()


# ── Manual Attendance ───────────────────────────────────────────────
@router.post("/attendance/manual", response_model=AttendanceLogOut)
async def mark_manual_attendance(
    payload: ManualAttendanceCreate,
    db: AsyncSession = Depends(get_db),
    manager: Employee = Depends(require_role("manager", "admin")),
):
    from app.models.models import DeviceBinding
    from datetime import datetime

    # Check employee exists
    emp_result = await db.execute(select(Employee).where(Employee.id == payload.employee_id))
    emp = emp_result.scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Check if log already exists for that date
    existing = await db.execute(
        select(AttendanceLog).where(
            and_(
                AttendanceLog.employee_id == payload.employee_id,
                AttendanceLog.date == payload.date,
            )
        )
    )
    log = existing.scalar_one_or_none()

    status_map = {
        "present": AttendanceStatus.PRESENT,
        "absent": AttendanceStatus.ABSENT,
        "exception": AttendanceStatus.EXCEPTION,
    }
    att_status = status_map.get(payload.status, AttendanceStatus.PRESENT)

    if log:
        old_status = log.status.value
        log.status = att_status
        log.notes = (log.notes or "") + f" [Manual: {payload.notes}]"
        await log_event(
            db, manager.id, "manual_attendance", "attendance_log", log.id,
            old_status, att_status.value, payload.notes,
        )
    else:
        log = AttendanceLog(
            employee_id=payload.employee_id,
            date=payload.date,
            status=att_status,
            notes=f"[Manual: {payload.notes}]",
        )
        if att_status == AttendanceStatus.PRESENT:
            log.check_in_time = datetime.utcnow()
        db.add(log)
        await db.flush()
        await db.refresh(log)
        await log_event(
            db, manager.id, "manual_attendance", "attendance_log", log.id,
            None, att_status.value, payload.notes,
        )

    return AttendanceLogOut(
        id=log.id,
        employee_id=log.employee_id,
        employee_name=emp.name,
        employee_code=emp.employee_id,
        date=log.date,
        check_in_time=log.check_in_time,
        check_out_time=log.check_out_time,
        confidence_score=log.confidence_score,
        method=log.method.value if log.method else "face",
        status=log.status.value,
        device_id=log.device_id,
        location_lat=log.location_lat,
        location_lng=log.location_lng,
        notes=log.notes or "",
        created_at=log.created_at,
    )


# ── Employee Devices ────────────────────────────────────────────────
@router.get("/devices")
async def list_employee_devices(
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    from app.models.models import DeviceBinding

    result = await db.execute(
        select(DeviceBinding).where(DeviceBinding.is_active == True)
    )
    bindings = result.scalars().all()

    # Batch-fetch employees to avoid N+1
    emp_ids = list({b.employee_id for b in bindings})
    emp_map = {}
    if emp_ids:
        emp_result = await db.execute(select(Employee).where(Employee.id.in_(emp_ids)))
        emp_map = {e.id: e for e in emp_result.scalars().all()}

    devices = []
    for b in bindings:
        emp = emp_map.get(b.employee_id)
        devices.append({
            "id": b.id,
            "employee_id": b.employee_id,
            "employee_name": emp.name if emp else "Unknown",
            "device_id": b.device_id,
            "device_name": b.device_name,
            "bound_at": b.bound_at.isoformat() if b.bound_at else None,
        })
    return devices

