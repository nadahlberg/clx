import json
import random
from io import StringIO

import pandas as pd
from django.conf import settings as django_settings
from django.contrib.postgres.indexes import GinIndex
from django.db import models
from django.utils import timezone
from django_shortuuid.fields import ShortUUIDField
from shortuuid import uuid

from clx.utils import generate_hash

from .search import SearchManager


class Base(models.Model):
    """Abstract base model for all CLX models."""

    id = ShortUUIDField(primary_key=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(default=timezone.now)

    objects = SearchManager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)


class Project(Base):
    """Model for projects."""

    name = models.CharField(max_length=255)
    instructions = models.TextField(blank=True, default="")
    active_label = models.ForeignKey(
        "Label",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )

    def add_docs(self, docs, **kwargs):
        """Bulk-insert documents using django-postgres-copy.

        Args:
            docs: Either a DataFrame with a 'text' column, a list of strings
                  (text only), or a list of dicts with 'text' and optionally
                  'meta' keys.
        """
        if docs is None:
            return

        # Normalize input
        if isinstance(docs, pd.DataFrame):
            if docs.empty:
                return
            if "text" not in docs.columns:
                raise ValueError(
                    "DataFrame input must include a 'text' column"
                )

            meta_columns = [col for col in docs.columns if col != "text"]
            docs = [
                {
                    "text": row["text"],
                    "meta": {
                        key: value
                        for key, value in row[meta_columns].items()
                        if pd.notna(value)
                    },
                }
                for _, row in docs.iterrows()
            ]
        elif not docs:
            return
        elif isinstance(docs[0], str):
            docs = [{"text": t, "meta": {}} for t in docs]
        else:
            docs = [
                {"text": d["text"], "meta": d.get("meta", {})} for d in docs
            ]

        # Build DataFrame
        data = pd.DataFrame(docs)
        data = data.dropna(subset=["text"])
        if len(data) == 0:
            return
        data["id"] = [uuid() for _ in range(len(data))]
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
        print("Pushing documents to database...")
        Document.objects.from_csv(
            f,
            static_mapping={
                "project_id": str(self.id),
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
        constraints = [
            models.UniqueConstraint(
                fields=["project", "text_hash"],
                name="document_project_text_hash_uniq",
            )
        ]
        indexes = [
            models.Index(
                fields=["shuffle_key", "id"],
                name="shuffle_key_idx",
            ),
            models.Index(
                fields=["text_prefix"],
                name="text_prefix_idx",
                opclasses=["text_pattern_ops"],
            ),
            GinIndex(
                fields=["text"],
                name="text_trgm_idx",
                opclasses=["gin_trgm_ops"],
            ),
        ]


class Label(Base):
    """Model for labels within a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="labels"
    )
    name = models.CharField(max_length=255)
    instructions = models.TextField(blank=True, default="")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="label_project_name_uniq",
            )
        ]


class Thread(Base):
    """Model for LLM threads tied to a label."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="threads"
    )
    model = models.CharField(
        max_length=255, default=django_settings.DEFAULT_MODEL
    )


class Message(Base):
    """Model for messages within a thread."""

    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="messages"
    )
    data = models.JSONField(default=dict)
    num_tokens = models.IntegerField(default=0)
