import json

from django.conf import settings as django_settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import Label, Project, Thread

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
    """Return a queryset of documents filtered by request params."""
    documents = project.documents.order_by("shuffle_key")
    qs = request.GET.get("q", "").strip()
    if qs:
        documents = documents.query_string(qs)
    label_id = request.GET.get("label", "").strip()
    if label_id:
        documents = documents.training_examples(label_id)
    return documents


@require_GET
def project_docs_api(request, project_id):
    """GET: paginated documents for a project."""
    project = get_object_or_404(Project, id=project_id)
    page_number = max(1, int(request.GET.get("page", 1)))
    documents = _filtered_documents(project, request)
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
    documents = _filtered_documents(project, request)
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
        latest = t.messages.order_by("-created_at").first()
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


@require_GET
def thread_messages_api(request, project_id, thread_id):
    """GET: list messages for a thread."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    messages = thread.messages.order_by("created_at")
    return JsonResponse(
        {
            "messages": [
                {
                    "id": str(m.id),
                    "data": m.data,
                    "num_tokens": m.num_tokens,
                }
                for m in messages
            ]
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
    return JsonResponse({"messages": new_messages})
