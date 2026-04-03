from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .models import (
    Project,
)


# Endpoints
@require_GET
def get_projects_endpoint(request):
    projects = Project.objects.all().order_by("name")
    return JsonResponse({"projects": projects})


# Views
def projects_view(request):
    return render(request, "projects.html")
