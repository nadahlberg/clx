import uuid

from django.db import models


class Project(models.Model):
    """Model for projects."""

    # use uuid id
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
