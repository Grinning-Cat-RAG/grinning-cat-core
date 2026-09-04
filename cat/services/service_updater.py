from typing import Dict

from cat.db import models
from cat.db.cruds import settings as crud_settings
from cat.log import log
from cat.services.service_factory import ServiceFactory
from cat.services.string_crypto import StringCrypto
from cat.utils import UpdaterFactory, SUFFIX_TO_CRYPT


class ServiceUpdater:
    def __init__(self, factory: ServiceFactory):
        self._factory = factory

    async def _update_factory_object(self, settings_name: str, settings: Dict) -> UpdaterFactory:
        """
        Update the settings of the factory object. This method upserts the settings in the database and returns
        an UpdaterFactory object containing the old and new settings.

        Args:
            settings_name: name of the settings
            settings: settings to update

        Returns:
            UpdaterFactory: object containing old and new settings
        """
        current_setting = await crud_settings.get_settings_by_category(self._factory.agent_key, self._factory.setting_category)

        # upsert the settings for the factory
        crypto = StringCrypto()
        final_setting = await crud_settings.upsert_setting_by_category(self._factory.agent_key, models.Setting(
            name=settings_name,
            category=self._factory.setting_category,
            value={
                k: crypto.encrypt(v)
                if isinstance(v, str) and any(suffix in k for suffix in SUFFIX_TO_CRYPT)
                else v
                for k, v in settings.items()
            },
        ))

        return UpdaterFactory(old_setting=current_setting, new_setting=final_setting)

    async def replace_service(self, model_provider_name: str, settings: Dict) -> Dict:
        """
        Replace the current service with a new one based on the provided name and settings.

        Args:
            model_provider_name: name of the new Service
            settings: settings of the new Service

        Returns:
            The dictionary resuming the new name and settings of the LLM
        """
        updater = None
        try:
            # Let plugins validate/enrich the settings payload BEFORE it is stored
            # (e.g. OpenRouter model validation and cost auto-population).
            # No-op by default; see base_plugin.hooks.factory.before_llm_settings_update.
            if self._factory.setting_category == "llm":
                enriched = await self._factory.hook_manager.execute_hook(
                    "before_llm_settings_update",
                    settings,
                    model_provider_name,
                    caller=None,
                )
                if enriched is not None:
                    settings = enriched

            updater = await self._update_factory_object(model_provider_name, settings)

            return {"name": model_provider_name, "value": updater.new_setting["value"]}  # type: ignore
        except Exception as e:
            log.error(f"Agent id: {self._factory.agent_key}. Error while loading the new Service: {e}")

            # something went wrong: rollback
            if updater is not None and updater.old_setting is not None:
                await self.replace_service(updater.old_setting["name"], updater.old_setting["value"])

            raise e
