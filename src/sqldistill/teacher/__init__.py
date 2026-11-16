"""Teachers: models that write question/SQL pairs for a schema."""

from __future__ import annotations

from sqldistill.teacher.base import Teacher
from sqldistill.teacher.fake import FakeTeacher

__all__ = ["FakeTeacher", "Teacher"]
