import json

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import Project

DOCS_PER_PAGE = 100

# --- Page Views ---


def projects_view(request):
    """Projects grid page."""
    projects = Project.objects.all().order_by("name")
    return render(request, "projects.html", {"projects": projects})


def project_detail_view(request, project_id):
    """Project detail page with paginated documents."""
    project = get_object_or_404(Project, id=project_id)
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
    text_filter = request.GET.get("text", "").strip()
    if text_filter:
        documents = documents.filter(text__icontains=text_filter)
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
