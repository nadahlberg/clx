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
    autopilot_enabled = models.BooleanField(default=False)
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

    def update_tasks(self):
        """Sync tasks based on current project/label state.

        Rules:
        - No project instructions → project_understanding task (no label)
        - Has project instructions but label lacks instructions → label_understanding per label
        - Both have instructions but label has no training examples → sampling_strategy per label
        - Label has unannotated training examples → annotate per label
        """
        from django.db.models import Count, Q

        expected = []  # list of (prompt_id, label_id | None)

        if not self.instructions.strip():
            expected.append(("project_understanding", None))
        else:
            labels = list(self.labels.all())
            for label in labels:
                if not label.instructions.strip():
                    expected.append(("label_understanding", label.id))
                else:
                    ld_stats = LabelDocument.objects.filter(label=label).aggregate(
                        total=Count("id"),
                        annotated=Count(
                            "id",
                            filter=Q(annotations__source="agent"),
                        ),
                    )
                    if ld_stats["total"] == 0:
                        expected.append(("sampling_strategy", label.id))
                    elif ld_stats["annotated"] < ld_stats["total"]:
                        expected.append(("annotate", label.id))

        expected_set = set(expected)
        existing = {
            (t.prompt_id, t.label_id): t
            for t in self.tasks.all()
        }

        # Delete tasks no longer expected
        to_delete = [
            t.id for key, t in existing.items() if key not in expected_set
        ]
        if to_delete:
            Task.objects.filter(id__in=to_delete).delete()

        # Create missing tasks
        to_create = [
            Task(project=self, prompt_id=pid, label_id=lid)
            for pid, lid in expected
            if (pid, lid) not in existing
        ]
        if to_create:
            Task.objects.bulk_create(to_create, ignore_conflicts=True)

        return list(
            self.tasks.select_related("label").order_by("created_at")
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
    autopilot_thread = models.ForeignKey(
        "Thread",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="label_project_name_uniq",
            )
        ]


class Prompt(Base):
    """A customizable prompt template for a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="prompts"
    )
    prompt_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    content = models.TextField(blank=True, default="")
    built_in = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "prompt_id"],
                name="prompt_project_promptid_uniq",
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
    state = models.JSONField(default=dict, blank=True)
    total_cost = models.FloatField(default=0.0)


class LabelDocument(Base):
    """Links a document to a label (e.g. as a training example)."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="label_documents"
    )
    document = models.ForeignKey(
        "Document", on_delete=models.CASCADE, related_name="label_documents"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["label", "document"],
                name="labeldocument_label_document_uniq",
            )
        ]


class ClassificationAnnotation(Base):
    """An annotation on a label document."""

    class Value(models.TextChoices):
        YES = "yes"
        NO = "no"
        SKIP = "skip"

    label_document = models.ForeignKey(
        LabelDocument, on_delete=models.CASCADE, related_name="annotations"
    )
    value = models.CharField(max_length=4, choices=Value.choices)
    source = models.CharField(max_length=255)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["label_document", "source"],
                name="annotation_labeldoc_source_uniq",
            )
        ]


class Task(Base):
    """A pending task for a project (e.g. 'annotate label X')."""

    class Status(models.TextChoices):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="tasks"
    )
    prompt_id = models.CharField(max_length=255)
    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, null=True, blank=True, related_name="tasks"
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "prompt_id", "label"],
                name="task_project_prompt_label_uniq",
            )
        ]


class Message(Base):
    """Model for messages within a thread."""

    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="messages"
    )
    data = models.JSONField(default=dict)
    num_tokens = models.IntegerField(default=0)
    is_compact = models.BooleanField(default=False)
