from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, EmailStr


# ── Auth ────────────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    employee_id: str
    name: str


class TokenData(BaseModel):
    sub: Optional[int] = None
    role: Optional[str] = None


# ── Employee ────────────────────────────────────────────────────────
class EmployeeCreate(BaseModel):
    employee_id: str
    name: str
    email: EmailStr
    password: str
    department: str = ""
    role: str = "employee"


class EmployeeLogin(BaseModel):
    email: str
    password: str


class EmployeeOut(BaseModel):
    id: int
    employee_id: str
    name: str
    email: str
    role: str
    department: str
    is_active: bool
    has_face_enrolled: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ── Device Binding ──────────────────────────────────────────────────
class DeviceBindingCreate(BaseModel):
    device_id: str
    device_name: str = ""


class DeviceBindingOut(BaseModel):
    id: int
    employee_id: int
    device_id: str
    device_name: str
    is_active: bool
    bound_at: datetime
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Attendance ──────────────────────────────────────────────────────
class CheckInRequest(BaseModel):
    device_id: str
    method: str = "face"  # "face" or "location"
    confidence_score: Optional[float] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None


class CheckOutRequest(BaseModel):
    device_id: str
    method: str = "face"
    confidence_score: Optional[float] = None


class AttendanceLogOut(BaseModel):
    id: int
    employee_id: int
    employee_name: str = ""
    employee_code: str = ""
    date: date
    check_in_time: Optional[datetime] = None
    check_out_time: Optional[datetime] = None
    confidence_score: Optional[float] = None
    method: str
    status: str
    device_id: Optional[str] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    notes: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


class AttendanceUpdate(BaseModel):
    check_in_time: Optional[datetime] = None
    check_out_time: Optional[datetime] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    reason: str = ""


# ── Photo Evidence ──────────────────────────────────────────────────
class PhotoEvidenceOut(BaseModel):
    id: int
    attendance_log_id: int
    image_path: str
    capture_time: datetime
    confidence_score: Optional[float] = None
    is_low_confidence: bool

    class Config:
        from_attributes = True


# ── Policy Config ───────────────────────────────────────────────────
class PolicyConfigOut(BaseModel):
    id: int
    key: str
    value: str
    description: str
    updated_at: datetime

    class Config:
        from_attributes = True


class PolicyConfigUpdate(BaseModel):
    value: str
    description: Optional[str] = None


# ── Audit Event ─────────────────────────────────────────────────────
class AuditEventOut(BaseModel):
    id: int
    actor_id: int
    action: str
    target_type: str
    target_id: Optional[int] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    reason: str
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Dashboard Summary ──────────────────────────────────────────────
class DashboardSummary(BaseModel):
    total_employees: int
    present_today: int
    absent_today: int
    not_checked_out: int
    exceptions: int
    low_confidence_count: int
