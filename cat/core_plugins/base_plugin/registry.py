import random
from typing import List
import httpx

from cat import PluginRegistry
from cat.log import log
from cat.looking_glass.models import PluginManifest
from cat.utils import write_temp_file


class CheshireCatPluginRegistry(PluginRegistry):
    @property
    def registry_url(self) :
        return "https://grinning-cat-plugins-backend-latest.onrender.com"

    async def search_plugins(self, query: str = None) -> List[PluginManifest]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # search plugins if a query is provided, list plugins (no search) otherwise
                response = await (
                    client.post(f"{self.registry_url}/search", json={"query": query})
                    if query
                    else client.get(f"{self.registry_url}/plugins", params={"page": 1, "page_size": 1000})
                )

                response.raise_for_status()
                json_response = response.json()

                plugins = json_response if query else json_response["plugins"]

            manifests = []
            for r in plugins:
                r["id"] = r["url"]
                manifests.append(PluginManifest(**r))

            # TODO: registry should sort plugins by score, until then we sort here at random
            random.shuffle(manifests)
            return manifests
        except httpx.ReadTimeout as e:
            log.warning(f"Timeout while calling plugins registry: {e}")
            return []
        except Exception as e:
            log.error(f"Error while calling plugins registry: {e}")
            return []

    async def download_plugin(self, url: str) -> str:
        log.info(f"Downloading {url}")

        payload = {"url": url}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.registry_url}/download", json=payload)
            response.raise_for_status()
            plugin_zip_path = await write_temp_file(f"{url.split('/')[-1]}.zip", response.content)

        log.info(f"Saved plugin as {plugin_zip_path}")
        return plugin_zip_path
