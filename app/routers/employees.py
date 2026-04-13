"""
Employee management router — device binding, face enrollment, employee CRUD (manager).
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.database import get_db
from app.models.models import Employee, DeviceBinding
from app.schemas.schemas import (
    DeviceBindingCreate, DeviceBindingOut, EmployeeOut, EmployeeUpdate,
)
from app.utils.auth import get_current_user, require_role
from app.services.audit_service import log_event
from app.services.face_service import generate_face_encoding

router = APIRouter(prefix="/api/employees", tags=["Employees"])


# ── Device Binding (self-service) ───────────────────────────────────
@router.post("/device/bind", response_model=DeviceBindingOut)
async def bind_device(
    payload: DeviceBindingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    # Check if employee already has an active binding
    existing = await db.execute(
        select(DeviceBinding).where(
            and_(
                DeviceBinding.employee_id == current_user.id,
                DeviceBinding.is_active == True,
            )
        )
    )
    active = existing.scalar_one_or_none()
    if active:
        raise HTTPException(
            status_code=400,
            detail="Already bound to a device. Contact your manager to rebind.",
        )

    # Check if device is already bound to another employee
    device_check = await db.execute(
        select(DeviceBinding).where(
            and_(
                DeviceBinding.device_id == payload.device_id,
                DeviceBinding.is_active == True,
            )
        )
    )
    if device_check.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Device already bound to another employee")

    binding = DeviceBinding(
        employee_id=current_user.id,
        device_id=payload.device_id,
        device_name=payload.device_name,
    )
    db.add(binding)
    await db.flush()
    await db.refresh(binding)

    await log_event(db, current_user.id, "bind_device", "device_binding", binding.id)
    return binding


@router.get("/device/status", response_model=DeviceBindingOut | None)
async def get_device_status(
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    result = await db.execute(
        select(DeviceBinding).where(
            and_(
                DeviceBinding.employee_id == current_user.id,
                DeviceBinding.is_active == True,
            )
        )
    )
    return result.scalar_one_or_none()


# ── Face Enrollment (self-service) ──────────────────────────────────
@router.post("/face/enroll")
async def enroll_face(
    photo: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    image_bytes = await photo.read()
    try:
        encoding = generate_face_encoding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Face enrollment failed. Please try again with a clear, well-lit selfie.",
        )
    current_user.face_encoding = encoding
    await db.flush()

    await log_event(db, current_user.id, "face_enroll", "employee", current_user.id)
    return {"message": "Face enrolled successfully"}


# ── Manager: list all employees ─────────────────────────────────────
@router.get("/", response_model=list[EmployeeOut])
async def list_employees(
    db: AsyncSession = Depends(get_db),
    _manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(select(Employee).order_by(Employee.name))
    employees = result.scalars().all()
    return [
        EmployeeOut(
            id=e.id,
            employee_id=e.employee_id,
            name=e.name,
            email=e.email,
            role=e.role.value,
            department=e.department,
            is_active=e.is_active,
            has_face_enrolled=e.face_encoding is not None,
            created_at=e.created_at,
        )
        for e in employees
    ]


# ── Manager: update employee ────────────────────────────────────────
@router.put("/{employee_id}", response_model=EmployeeOut)
async def update_employee(
    employee_id: int,
    payload: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
    manager: Employee = Depends(require_role("manager", "admin")),
):
    result = await db.execute(select(Employee).where(Employee.id == employee_id))
    emp = result.scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    old_values = {}
    for field, val in payload.model_dump(exclude_unset=True).items():
        old_values[field] = getattr(emp, field)
        setattr(emp, field, val)

    await log_event(
        db, manager.id, "update_employee", "employee", emp.id,
        str(old_values), str(payload.model_dump(exclude_unset=True)),
    )

    return EmployeeOut(
        id=emp.id,
        employee_id=emp.employee_id,
        name=emp.name,
        email=emp.email,
        role=emp.role.value if hasattr(emp.role, "value") else emp.role,
        department=emp.department,
        is_active=emp.is_active,
        has_face_enrolled=emp.face_encoding is not None,
        created_at=emp.created_at,
    )


# ── Manager: rebind device ──────────────────────────────────────────
@router.post("/{employee_id}/device/rebind", response_model=DeviceBindingOut)
async def rebind_device(
    employee_id: int,
    payload: DeviceBindingCreate,
    db: AsyncSession = Depends(get_db),
    manager: Employee = Depends(require_role("manager", "admin")),
):
    # Revoke existing binding
    result = await db.execute(
        select(DeviceBinding).where(
            and_(
                DeviceBinding.employee_id == employee_id,
                DeviceBinding.is_active == True,
            )
        )
    )
    old = result.scalar_one_or_none()
    if old:
        old.is_active = False
        old.revoked_at = datetime.utcnow()

    binding = DeviceBinding(
        employee_id=employee_id,
        device_id=payload.device_id,
        device_name=payload.device_name,
    )
    db.add(binding)
    await db.flush()
    await db.refresh(binding)

    await log_event(db, manager.id, "rebind_device", "device_binding", binding.id,
                    reason="Manager override")
    return binding
