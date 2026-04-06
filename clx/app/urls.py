from django.urls import path

from . import views

urlpatterns = [
    path("", views.projects_view),
    path(
        "projects/<str:project_id>/",
        views.project_detail_view,
    ),
    path("api/projects/", views.projects_api),
    path(
        "api/projects/<str:project_id>/docs",
        views.project_docs_api,
    ),
]
