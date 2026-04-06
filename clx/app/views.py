import json

from django.core.paginator import Paginator
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


@require_GET
def project_docs_api(request, project_id):
    """GET: paginated documents for a project."""
    project = get_object_or_404(Project, id=project_id)
    page_number = request.GET.get("page", 1)
    documents = project.documents.order_by("shuffle_key")
    paginator = Paginator(documents, DOCS_PER_PAGE)
    page = paginator.get_page(page_number)
    return JsonResponse(
        {
            "documents": [
                {
                    "id": str(d.id),
                    "text": d.text,
                    "text_prefix": d.text_prefix,
                    "meta": d.meta,
                }
                for d in page
            ],
            "page": page.number,
            "total_pages": paginator.num_pages,
            "total_count": paginator.count,
        }
    )
