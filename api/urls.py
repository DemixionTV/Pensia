from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import CSVProcessViewSet

router = DefaultRouter()
router.register(r'create-prediction', CSVProcessViewSet, basename='predict')

urlpatterns = [
    path('v1/', include(router.urls))
]
