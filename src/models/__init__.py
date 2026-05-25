"""
Re-export student and teacher model classes for convenient imports.

    from src.models import TeacherModel, StudentModel, MockTeacherModel, MockStudentModel
"""
from src.models.teacher import TeacherModel, MockTeacherModel
from src.models.student import StudentModel, MockStudentModel, COMPRESSION_VARIANTS

__all__ = [
    "TeacherModel",
    "MockTeacherModel",
    "StudentModel",
    "MockStudentModel",
    "COMPRESSION_VARIANTS",
]
