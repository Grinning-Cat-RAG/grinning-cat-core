import pytest

from cat.services.contextpress_service import ContextPressService


def test_contextpress_service_available():
    service = ContextPressService()
    # If contpress isn't installed, the service reports not available — skip in that case
    if not service.available():
        pytest.skip("contpress not installed in test environment")

    # Basic smoke checks
    assert service.model is not None
    assert isinstance(service.count("hello"), int)
