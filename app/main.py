"""
Native Attendance Application — FastAPI Backend
"""
import os
import asyncio
from datetime import datetime, date, time, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, and_

from app.database import init_db, async_session
from app.config import get_settings
from app.models.models import (
    Employee, AttendanceLog, PolicyConfig,
    UserRole, AttendanceStatus,
)
from app.utils.auth import hash_password
from app.routers import auth, attendance, employees, manager, policy

settings = get_settings()


# ── Auto-seed test accounts ─────────────────────────────────────────
async def seed_accounts():
    """Create default test accounts if they don't exist."""
    accounts = [
        {
            "employee_id": "ADM-001",
            "name": "Admin User",
            "email": "admin@test.com",
            "password": "test1234",
            "department": "Administration",
            "role": UserRole.ADMIN,
        },
        {
            "employee_id": "MGR-001",
            "name": "Manager User",
            "email": "manager@test.com",
            "password": "test1234",
            "department": "Management",
            "role": UserRole.MANAGER,
        },
        {
            "employee_id": "EMP-101",
            "name": "Employee User",
            "email": "employee@test.com",
            "password": "test1234",
            "department": "Engineering",
            "role": UserRole.EMPLOYEE,
        },
    ]
    async with async_session() as db:
        for acc in accounts:
            result = await db.execute(
                select(Employee).where(Employee.email == acc["email"])
            )
            if result.scalar_one_or_none() is None:
                emp = Employee(
                    employee_id=acc["employee_id"],
                    name=acc["name"],
                    email=acc["email"],
                    password_hash=hash_password(acc["password"]),
                    department=acc["department"],
                    role=acc["role"],
                )
                db.add(emp)
        await db.commit()


# ── Scheduled tasks ─────────────────────────────────────────────────
async def run_daily_tasks():
    """Background loop: run auto-absent and auto-checkout once per minute check."""
    while True:
        try:
            now = datetime.utcnow()
            async with async_session() as db:
                # Read cutoff_time from policy (default 18:00)
                result = await db.execute(
                    select(PolicyConfig).where(PolicyConfig.key == "cutoff_time")
                )
                policy = result.scalar_one_or_none()
                cutoff_str = policy.value if policy else "18:00"
                cutoff_h, cutoff_m = map(int, cutoff_str.split(":"))
                cutoff_time = time(cutoff_h, cutoff_m)

                # Only run after cutoff time
                if now.time() >= cutoff_time:
                    today = date.today()

                    # Auto-mark NOT_CHECKED_OUT for employees who checked in but not out
                    logs = await db.execute(
                        select(AttendanceLog).where(
                            and_(
                                AttendanceLog.date == today,
                                AttendanceLog.status == AttendanceStatus.PRESENT,
                                AttendanceLog.check_out_time.is_(None),
                            )
                        )
                    )
                    for log in logs.scalars().all():
                        log.status = AttendanceStatus.NOT_CHECKED_OUT
                        log.check_out_time = datetime.combine(today, cutoff_time)
                        log.notes = (log.notes or "") + " [Auto-checkout at cutoff]"

                    # Auto-mark ABSENT for active employees with no log today
                    all_employees = await db.execute(
                        select(Employee).where(Employee.is_active == True)
                    )
                    for emp in all_employees.scalars().all():
                        existing = await db.execute(
                            select(AttendanceLog).where(
                                and_(
                                    AttendanceLog.employee_id == emp.id,
                                    AttendanceLog.date == today,
                                )
                            )
                        )
                        if existing.scalar_one_or_none() is None:
                            absent_log = AttendanceLog(
                                employee_id=emp.id,
                                date=today,
                                status=AttendanceStatus.ABSENT,
                                notes="[Auto-marked absent]",
                            )
                            db.add(absent_log)

                    await db.commit()
        except Exception:
            pass  # Don't crash the background task

        await asyncio.sleep(300)  # Check every 5 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, upload dir, seed accounts
    await init_db()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    await seed_accounts()
    # Start background scheduler
    task = asyncio.create_task(run_daily_tasks())
    yield
    # Shutdown
    task.cancel()


app = FastAPI(
    title="Native Attendance API",
    description="Face-based attendance system with device binding, location fallback, and manager controls",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter app + web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads for photo serving
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR, check_dir=False), name="uploads")

# Register routers
app.include_router(auth.router)
app.include_router(attendance.router)
app.include_router(employees.router)
app.include_router(manager.router)
app.include_router(policy.router)


@app.get("/")
async def root():
    return {
        "app": "Native Attendance API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}

