"""The ingestion profile bundle on the configuration.

The configuration class is the single place that knows which implementation
owns each ingestion part; ``profile()`` instantiates them and
``get_from_config`` attaches the bundle to the engine it builds. These tests
pin the core bundle and the attachment, and that the factory validation is
still the base one.
"""

import pytest

from cat.services.factory.ingestion import (
    BaseIngestionConfiguration,
    BaseIngestionEngine,
    CoreIngestionConfiguration,
    CoreIngestionEngine,
)
from cat.services.ingestion.locks import NoopLockManager
from cat.services.ingestion.pipeline import CoreSinglePassPipeline
from cat.services.ingestion.profile import IngestionProfile
from cat.services.ingestion.recovery import NoopRecovery
from cat.services.ingestion.state import CoreIngestionStateStore
from cat.services.ingestion.tags import SettingsVersionTags


def test_core_profile_returns_the_five_core_implementations():
    profile = CoreIngestionConfiguration.profile()

    assert isinstance(profile, IngestionProfile)
    assert isinstance(profile.pipeline, CoreSinglePassPipeline)
    assert isinstance(profile.state_store, CoreIngestionStateStore)
    assert isinstance(profile.recovery, NoopRecovery)
    assert isinstance(profile.locks, NoopLockManager)
    assert isinstance(profile.tags, SettingsVersionTags)


def test_core_configuration_classmethods_return_the_core_classes():
    assert CoreIngestionConfiguration.pipeline() is CoreSinglePassPipeline
    assert CoreIngestionConfiguration.state_store() is CoreIngestionStateStore
    assert CoreIngestionConfiguration.recovery() is NoopRecovery
    assert CoreIngestionConfiguration.locks() is NoopLockManager
    assert CoreIngestionConfiguration.tags() is SettingsVersionTags


def test_get_from_config_attaches_a_non_none_profile_to_the_engine():
    engine = CoreIngestionConfiguration.get_from_config({})

    assert isinstance(engine, CoreIngestionEngine)
    assert engine.profile is not None
    assert isinstance(engine.profile.pipeline, CoreSinglePassPipeline)
    assert isinstance(engine.profile.state_store, CoreIngestionStateStore)
    assert isinstance(engine.profile.recovery, NoopRecovery)
    assert isinstance(engine.profile.locks, NoopLockManager)
    assert isinstance(engine.profile.tags, SettingsVersionTags)


def test_profile_is_a_new_instance_per_engine_never_a_singleton():
    first = CoreIngestionConfiguration.get_from_config({})
    second = CoreIngestionConfiguration.get_from_config({})

    assert first.profile is not None and second.profile is not None
    assert first.profile is not second.profile


def test_engine_built_directly_has_no_profile():
    assert CoreIngestionEngine().profile is None


def test_get_from_config_keeps_the_pyclass_base_class_validation():
    class InvalidIngestionEngine:
        pass

    class InvalidIngestionConfiguration(BaseIngestionConfiguration):
        @classmethod
        def pyclass(cls) -> type:
            return InvalidIngestionEngine

    with pytest.raises(Exception, match="valid BaseIngestionEngine class"):
        InvalidIngestionConfiguration.get_from_config({})


def test_profile_attaches_to_a_plugin_engine_too():
    class OtherEngine(BaseIngestionEngine):
        async def run(self, lizard) -> bool:
            return True

        async def ingest_file(self, cat, file, filename=None, metadata=None, store_file=True, content_type=None) -> None:
            pass

    class OtherConfiguration(BaseIngestionConfiguration):
        @classmethod
        def pyclass(cls) -> type:
            return OtherEngine

    engine = OtherConfiguration.get_from_config({})

    assert isinstance(engine, OtherEngine)
    assert engine.profile is not None
    assert isinstance(engine.profile.pipeline, CoreSinglePassPipeline)
