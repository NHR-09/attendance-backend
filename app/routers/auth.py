"""
Auth router — register, login, profile.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.models import Employee
from app.schemas.schemas import EmployeeCreate, EmployeeLogin, EmployeeOut, TokenResponse
from app.utils.auth import hash_password, verify_password, create_access_token, get_current_user
from app.services.audit_service import log_event

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=EmployeeOut, status_code=201)
async def register(payload: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    # check duplicates
    existing = await db.execute(
        select(Employee).where(
            (Employee.email == payload.email) | (Employee.employee_id == payload.employee_id)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email or Employee ID already registered")

    # Security: always force role to EMPLOYEE on public registration.
    # Only admins can promote users via the update endpoint.
    from app.models.models import UserRole
    employee = Employee(
        employee_id=payload.employee_id,
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        department=payload.department,
        role=UserRole.EMPLOYEE,
    )
    db.add(employee)
    await db.flush()
    await db.refresh(employee)

    await log_event(db, employee.id, "register", "employee", employee.id)

    return EmployeeOut(
        id=employee.id,
        employee_id=employee.employee_id,
        name=employee.name,
        email=employee.email,
        role=employee.role.value if hasattr(employee.role, "value") else employee.role,
        department=employee.department,
        is_active=employee.is_active,
        has_face_enrolled=employee.face_encoding is not None,
        created_at=employee.created_at,
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: EmployeeLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.email == payload.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    role_val = user.role.value if hasattr(user.role, "value") else user.role
    token = create_access_token({"sub": str(user.id), "role": role_val})

    await log_event(db, user.id, "login", "employee", user.id)

    return TokenResponse(
        access_token=token,
        role=role_val,
        employee_id=user.employee_id,
        name=user.name,
    )


@router.get("/me", response_model=EmployeeOut)
async def get_profile(current_user: Employee = Depends(get_current_user)):
    return EmployeeOut(
        id=current_user.id,
        employee_id=current_user.employee_id,
        name=current_user.name,
        email=current_user.email,
        role=current_user.role.value,
        department=current_user.department,
        is_active=current_user.is_active,
        has_face_enrolled=current_user.face_encoding is not None,
        created_at=current_user.created_at,
    )
