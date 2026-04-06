import json

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import Label, Project

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
    return render(request, "project_detail.html", {"project": project})


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
                {"id": str(l.id), "name": l.name} for l in labels
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
    """POST: rename a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    label.name = name
    label.save(update_fields=["name", "updated_at"])
    return JsonResponse({"id": str(label.id), "name": label.name})


@csrf_exempt
@require_http_methods(["POST"])
def rename_project_api(request, project_id):
    """POST: rename a project."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    project.name = name
    project.save(update_fields=["name", "updated_at"])
    return JsonResponse({"id": str(project.id), "name": project.name})
