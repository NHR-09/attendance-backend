"""
Audit logging service — records every privileged action.
"""
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import AuditEvent


async def log_event(
    db: AsyncSession,
    actor_id: int,
    action: str,
    target_type: str,
    target_id: int | None = None,
    old_value: str | None = None,
    new_value: str | None = None,
    reason: str = "",
):
    event = AuditEvent(
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        timestamp=datetime.utcnow(),
    )
    db.add(event)
    await db.flush()
    return event
