import enum
from datetime import datetime, date, time
from sqlalchemy import (
    Column, Integer, String, Float, Date, Time, DateTime,
    Boolean, Text, Enum, ForeignKey, LargeBinary
)
from sqlalchemy.orm import relationship
from app.database import Base


class UserRole(str, enum.Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"


class AttendanceMethod(str, enum.Enum):
    FACE = "face"
    LOCATION = "location"


class AttendanceStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    NOT_CHECKED_OUT = "not_checked_out"
    EXCEPTION = "exception"


# ── Employee ────────────────────────────────────────────────────────
class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.EMPLOYEE, nullable=False)
    department = Column(String(100), default="")
    face_encoding = Column(LargeBinary, nullable=True)  # stored numpy bytes
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # relationships
    device_bindings = relationship("DeviceBinding", back_populates="employee", lazy="selectin")
    attendance_logs = relationship("AttendanceLog", back_populates="employee", lazy="selectin")


# ── Device Binding ──────────────────────────────────────────────────
class DeviceBinding(Base):
    __tablename__ = "device_bindings"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    device_id = Column(String(255), nullable=False, index=True)
    device_name = Column(String(255), default="")
    is_active = Column(Boolean, default=True)
    bound_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)

    employee = relationship("Employee", back_populates="device_bindings")


# ── Attendance Log ──────────────────────────────────────────────────
class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    confidence_score = Column(Float, nullable=True)
    method = Column(Enum(AttendanceMethod), default=AttendanceMethod.FACE)
    status = Column(Enum(AttendanceStatus), default=AttendanceStatus.PRESENT)
    device_id = Column(String(255), nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", back_populates="attendance_logs")
    photos = relationship("PhotoEvidence", back_populates="attendance_log", lazy="selectin")


# ── Photo Evidence ──────────────────────────────────────────────────
class PhotoEvidence(Base):
    __tablename__ = "photo_evidence"

    id = Column(Integer, primary_key=True, index=True)
    attendance_log_id = Column(Integer, ForeignKey("attendance_logs.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    capture_time = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, nullable=True)
    is_low_confidence = Column(Boolean, default=False)

    attendance_log = relationship("AttendanceLog", back_populates="photos")


# ── Policy Config ───────────────────────────────────────────────────
class PolicyConfig(Base):
    __tablename__ = "policy_configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text, default="")
    updated_by = Column(Integer, ForeignKey("employees.id"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Audit Event ─────────────────────────────────────────────────────
class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    action = Column(String(50), nullable=False, index=True)  # e.g. "check_in", "override", "edit"
    target_type = Column(String(50), nullable=False)  # e.g. "attendance_log", "employee"
    target_id = Column(Integer, nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    reason = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
