import json

from django.conf import settings as django_settings
from django.db.models import Count, Q, Sum
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import Label, Project, Prompt, Thread

DOCS_PER_PAGE = 100

# --- Page Views ---


def projects_view(request):
    """Projects grid page."""
    projects = Project.objects.all().order_by("name")
    return render(request, "projects.html", {"projects": projects})


def project_detail_view(request, project_id):
    """Project detail page with paginated documents."""
    project = get_object_or_404(Project, id=project_id)
    # Lazily create an initial label if none exist.
    if not project.labels.exists():
        label = Label.objects.create(project=project, name="Initial Label")
        project.active_label = label
        project.save(update_fields=["active_label", "updated_at"])
    elif project.active_label is None:
        project.active_label = project.labels.first()
        project.save(update_fields=["active_label", "updated_at"])
    return render(
        request,
        "project_detail.html",
        {
            "project": project,
            "model_ids": django_settings.MODEL_IDS,
            "default_model": django_settings.DEFAULT_MODEL,
        },
    )


# --- API Endpoints ---


@csrf_exempt
@require_http_methods(["GET", "POST"])
def projects_api(request):
    """GET: list projects. POST: create a project."""
    if request.method == "GET":
        projects = Project.objects.all().order_by("name")
        return JsonResponse(
            {"projects": [{"id": str(p.id), "name": p.name} for p in projects]}
        )
    else:
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        name = data.get("name", "").strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        project = Project.objects.create(name=name)
        return JsonResponse(
            {"id": str(project.id), "name": project.name},
            status=201,
        )


def _filtered_documents(project, request):
    """Return a queryset of documents filtered by request params.

    Returns (queryset, label_id, has_annotation_col).
    """
    documents = project.documents.order_by("shuffle_key")
    qs = request.GET.get("q", "").strip()
    if qs:
        documents = documents.query_string(qs)
    label_id = request.GET.get("label", "").strip()
    annotation = request.GET.get("annotation", "").strip()
    has_annotation_col = False
    if annotation and label_id:
        documents = documents.filter_annotation(label_id, annotation)
        has_annotation_col = True
    elif label_id:
        documents = documents.training_examples(label_id)
    return documents, label_id, has_annotation_col


@require_GET
def project_docs_api(request, project_id):
    """GET: paginated documents for a project."""
    project = get_object_or_404(Project, id=project_id)
    page_number = max(1, int(request.GET.get("page", 1)))
    documents, label_id, has_annotation_col = _filtered_documents(
        project, request
    )
    # Attach annotation values in a single subquery when a label is active.
    if label_id and not has_annotation_col:
        documents = documents.with_annotation(label_id)
    # Fetch one extra to detect if there's a next page, avoiding COUNT query.
    offset = (page_number - 1) * DOCS_PER_PAGE
    batch = list(documents[offset : offset + DOCS_PER_PAGE + 1])
    has_next = len(batch) > DOCS_PER_PAGE
    page_docs = batch[:DOCS_PER_PAGE]
    return JsonResponse(
        {
            "documents": [
                {
                    "id": str(d.id),
                    "text": d.text,
                    "text_prefix": d.text_prefix,
                    "meta": d.meta,
                    "annotation": getattr(d, "annotation_value", None),
                }
                for d in page_docs
            ],
            "page": page_number,
            "has_next": has_next,
        }
    )


@require_GET
def project_docs_count_api(request, project_id):
    """GET: count of documents for a project (with filters)."""
    project = get_object_or_404(Project, id=project_id)
    documents, _, _ = _filtered_documents(project, request)
    return JsonResponse({"total_count": documents.count()})


@require_GET
def project_labels_api(request, project_id):
    """GET: list labels for a project."""
    project = get_object_or_404(Project, id=project_id)
    labels = project.labels.order_by("name")
    return JsonResponse(
        {
            "labels": [
                {
                    "id": str(label.id),
                    "name": label.name,
                    "instructions": label.instructions,
                }
                for label in labels
            ],
            "active_label_id": str(project.active_label_id)
            if project.active_label_id
            else None,
        }
    )


@require_GET
def label_stats_api(request, project_id, label_id):
    """GET: annotation stats for a label's training set."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    from .models import LabelDocument

    stats = LabelDocument.objects.filter(label=label).aggregate(
        total=Count("id"),
        yes=Count(
            "id",
            filter=Q(annotations__value="yes", annotations__source="agent"),
        ),
        no=Count(
            "id",
            filter=Q(annotations__value="no", annotations__source="agent"),
        ),
        skip=Count(
            "id",
            filter=Q(annotations__value="skip", annotations__source="agent"),
        ),
    )
    stats["unannotated"] = (
        stats["total"] - stats["yes"] - stats["no"] - stats["skip"]
    )
    return JsonResponse(stats)


@csrf_exempt
@require_http_methods(["POST"])
def create_label_api(request, project_id):
    """POST: create a new label for a project."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    label = Label.objects.create(project=project, name=name)
    return JsonResponse({"id": str(label.id), "name": label.name}, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def set_active_label_api(request, project_id, label_id):
    """POST: set the active label for a project."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    project.active_label = label
    project.save(update_fields=["active_label", "updated_at"])
    return JsonResponse({"id": str(label.id), "name": label.name})


@csrf_exempt
@require_http_methods(["POST"])
def rename_label_api(request, project_id, label_id):
    """POST: update a label (name, instructions)."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "name" in data:
        name = data["name"].strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        label.name = name
        update_fields.append("name")
    if "instructions" in data:
        label.instructions = data["instructions"]
        update_fields.append("instructions")
    label.save(update_fields=update_fields)
    return JsonResponse(
        {
            "id": str(label.id),
            "name": label.name,
            "instructions": label.instructions,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def update_project_api(request, project_id):
    """POST: update project settings."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "name" in data:
        name = data["name"].strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        project.name = name
        update_fields.append("name")
    if "instructions" in data:
        project.instructions = data["instructions"]
        update_fields.append("instructions")
    project.save(update_fields=update_fields)
    return JsonResponse(
        {
            "id": str(project.id),
            "name": project.name,
            "instructions": project.instructions,
        }
    )


def _prompt_json(p):
    return {
        "id": str(p.id),
        "prompt_id": p.prompt_id,
        "name": p.name,
        "content": p.content,
        "built_in": p.built_in,
    }


def _slugify_prompt_id(name):
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@require_GET
def project_prompts_api(request, project_id):
    """GET: list prompts for a project, lazy-creating from registry."""
    from .prompts import prompt_registry

    project = get_object_or_404(Project, id=project_id)
    existing = {p.prompt_id for p in project.prompts.all()}
    to_create = [
        Prompt(
            project=project,
            prompt_id=pid,
            name=entry["name"],
            content=entry["content"],
            built_in=True,
        )
        for pid, entry in prompt_registry.items()
        if pid not in existing
    ]
    if to_create:
        Prompt.objects.bulk_create(to_create)
    prompts = project.prompts.order_by("built_in", "created_at")
    return JsonResponse({"prompts": [_prompt_json(p) for p in prompts]})


@csrf_exempt
@require_http_methods(["POST"])
def create_prompt_api(request, project_id):
    """POST: create a custom prompt."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    prompt = Prompt.objects.create(
        project=project,
        prompt_id=_slugify_prompt_id(name),
        name=name,
        content=data.get("content", ""),
        built_in=False,
    )
    return JsonResponse(_prompt_json(prompt), status=201)


@csrf_exempt
@require_http_methods(["POST"])
def update_prompt_api(request, project_id, prompt_id):
    """POST: update a prompt's content (and name for custom prompts)."""
    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "content" in data:
        prompt.content = data["content"]
        update_fields.append("content")
    if "name" in data and not prompt.built_in:
        name = data["name"].strip()
        if name:
            prompt.name = name
            prompt.prompt_id = _slugify_prompt_id(name)
            update_fields.extend(["name", "prompt_id"])
    prompt.save(update_fields=update_fields)
    return JsonResponse(_prompt_json(prompt))


@csrf_exempt
@require_http_methods(["POST"])
def reset_prompt_api(request, project_id, prompt_id):
    """POST: reset a prompt's content to the registry default."""
    from .prompts import prompt_registry

    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    default = prompt_registry.get(prompt.prompt_id)
    if not default:
        return JsonResponse(
            {"error": "No default found for this prompt."}, status=400
        )
    prompt.content = default["content"]
    prompt.save(update_fields=["content", "updated_at"])
    return JsonResponse(_prompt_json(prompt))


@csrf_exempt
@require_http_methods(["POST"])
def delete_prompt_api(request, project_id, prompt_id):
    """POST: delete a custom prompt."""
    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    if prompt.built_in:
        return JsonResponse(
            {"error": "Cannot delete a built-in prompt."}, status=400
        )
    prompt.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def delete_project_api(request, project_id):
    """POST: delete a project."""
    project = get_object_or_404(Project, id=project_id)
    project.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def delete_label_api(request, project_id, label_id):
    """POST: delete a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    if project.labels.count() <= 1:
        return JsonResponse(
            {"error": "Cannot delete the only label."}, status=400
        )
    was_active = project.active_label_id == label.id
    label.delete()
    if was_active:
        new_active = project.labels.first()
        project.active_label = new_active
        project.save(update_fields=["active_label", "updated_at"])
    return JsonResponse({"ok": True})


@require_GET
def label_threads_api(request, project_id, label_id):
    """GET: list threads for a label with latest message preview."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    threads = label.threads.order_by("-updated_at")
    result = []
    for t in threads:
        latest = t.messages.order_by("created_at").first()
        preview = ""
        if latest and latest.data.get("content"):
            preview = latest.data["content"][:100]
        result.append(
            {
                "id": str(t.id),
                "model": t.model,
                "preview": preview,
                "updated_at": t.updated_at.isoformat(),
            }
        )
    return JsonResponse({"threads": result})


@csrf_exempt
@require_http_methods(["POST"])
def create_thread_api(request, project_id, label_id):
    """POST: create a new thread for a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}
    model = data.get("model", django_settings.DEFAULT_MODEL)
    thread = Thread.objects.create(label=label, model=model)
    return JsonResponse(
        {"id": str(thread.id), "model": thread.model}, status=201
    )


def _active_token_count(thread, messages=None):
    """Sum num_tokens from the last compact point onward."""
    compact_msg = (
        thread.messages.filter(is_compact=True)
        .order_by("-created_at")
        .values_list("created_at", flat=True)
        .first()
    )
    if compact_msg:
        qs = thread.messages.filter(created_at__gte=compact_msg)
    else:
        qs = thread.messages
    return qs.aggregate(total=Sum("num_tokens"))["total"] or 0


@require_GET
def thread_messages_api(request, project_id, thread_id):
    """GET: list messages for a thread."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    messages = list(thread.messages.order_by("created_at"))
    total_tokens = _active_token_count(thread, messages)
    return JsonResponse(
        {
            "messages": [
                {
                    "id": str(m.id),
                    "data": m.data,
                    "num_tokens": m.num_tokens,
                    "is_compact": m.is_compact,
                }
                for m in messages
            ],
            "usage": {
                "total_tokens": total_tokens,
                "total_cost": thread.total_cost,
            },
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def delete_thread_api(request, project_id, thread_id):
    """POST: delete a thread."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    thread.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def send_message_api(request, project_id, thread_id):
    """POST: send a user message and get an agent response."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    content = data.get("content", "").strip()
    if not content:
        return JsonResponse({"error": "content is required"}, status=400)

    from .agent import CLXAgent

    agent = CLXAgent(thread)
    msg_count_before = len(agent.messages)
    agent.run(content)
    # Return all new messages (skipping the user message already shown)
    new_messages = [
        m
        for m in agent.messages[msg_count_before + 1 :]
        if m.get("role") != "system"
    ]
    thread.refresh_from_db(fields=["total_cost"])
    total_tokens = _active_token_count(thread)
    return JsonResponse(
        {
            "messages": new_messages,
            "usage": {
                "total_tokens": total_tokens,
                "total_cost": thread.total_cost,
            },
        }
    )
