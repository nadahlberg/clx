import json
import random
from io import StringIO

import pandas as pd
from django.db import models
from django.utils import timezone
from django_shortuuid.fields import ShortUUIDField
from postgres_copy import CopyManager

from clx.utils import generate_hash


class Base(models.Model):
    """Abstract base model for all CLX models."""

    id = ShortUUIDField(primary_key=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(default=timezone.now)

    objects = CopyManager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)


class Project(Base):
    """Model for projects."""

    name = models.CharField(max_length=255)

    def add_docs(self, docs, **kwargs):
        """Bulk-insert documents using django-postgres-copy.

        Args:
            docs: Either a list of strings (text only) or a list of
                  dicts with 'text' and optionally 'meta' keys.
        """
        if not docs:
            return

        # Normalize input
        if isinstance(docs[0], str):
            docs = [{"text": t, "meta": {}} for t in docs]
        else:
            docs = [
                {"text": d["text"], "meta": d.get("meta", {})} for d in docs
            ]

        # Build DataFrame
        data = pd.DataFrame(docs)
        data["text_prefix"] = data["text"].str[:50]
        data["text_hash"] = data["text"].apply(generate_hash)
        data["shuffle_key"] = [
            random.randint(0, 1_000_000) for _ in range(len(data))
        ]
        data["meta"] = data["meta"].apply(json.dumps)

        # Write to CSV buffer
        f = StringIO()
        data.to_csv(f, index=False)
        f.seek(0)

        # Bulk insert
        Document.objects.from_csv(
            f,
            static_mapping={
                "project": self.id,
                "created_at": timezone.now(),
                "updated_at": timezone.now(),
            },
            ignore_conflicts=True,
            **kwargs,
        )


class Document(Base):
    """Model for documents within a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="documents"
    )
    text = models.TextField()
    text_prefix = models.CharField(max_length=50)
    meta = models.JSONField(default=dict, null=True, blank=True)
    shuffle_key = models.IntegerField()
    text_hash = models.CharField(max_length=64)

    class Meta:
        unique_together = [("project", "text_hash")]
