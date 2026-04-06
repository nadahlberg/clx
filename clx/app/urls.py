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
    path(
        "api/projects/<str:project_id>/docs/count",
        views.project_docs_count_api,
    ),
    path(
        "api/projects/<str:project_id>/labels",
        views.project_labels_api,
    ),
    path(
        "api/projects/<str:project_id>/labels/create",
        views.create_label_api,
    ),
    path(
        "api/projects/<str:project_id>/labels/<str:label_id>/activate",
        views.set_active_label_api,
    ),
    path(
        "api/projects/<str:project_id>/labels/<str:label_id>/rename",
        views.rename_label_api,
    ),
    path(
        "api/projects/<str:project_id>/rename",
        views.rename_project_api,
    ),
]
