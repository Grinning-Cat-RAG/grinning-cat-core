import asyncio
import json
import random
import redis.asyncio as aioredis
from typing import Dict, Callable, Optional

from cat import log, hook
from cat.db.database import get_async_db
from cat.utils import pod_id, singleton

# Max messages to keep in the stream to prevent memory explosion
STREAM_MAX_LEN = 1000


class MarchHareConfig:
    # List of streams for event management
    streams = {
        "PLUGIN_EVENTS": "march_hare:streams:plugin_events"
    }

    # list of event types for plugin management
    events = {
        "PLUGIN_INSTALLATION": "plugin_installation",
        "PLUGIN_UNINSTALLATION": "plugin_uninstallation",
        # a plugin was toggled on a system level: the other PODs align their plugins with the database
        "PLUGIN_SYSTEM_TOGGLE": "plugin_system_toggle",
    }


@singleton
class MarchHare:
    def __init__(self):
        self.pod_id = pod_id()
        self._async_redis_client = get_async_db()
        self._stop_event = asyncio.Event()

        # Set once the lizard is fully bootstrapped; gates handler spawning so that
        # listener tasks can be started early (before_lizard_bootstrap) without
        # dispatching events to a not-yet-ready system.
        self._ready_event = asyncio.Event()

        # One long-running listener Task per stream (stream_name → Task).
        self._listeners: Dict[str, asyncio.Task] = {}

        # One short-lived handler Task per message being processed
        # (stream_name:message_id → Task).  Tasks are added when a message
        # arrives and removed automatically when they complete.
        self._handlers: Dict[str, asyncio.Task] = {}

        # Backoff settings (used as defaults per listener loop)
        self._base_delay = 1.0   # Start with 1 second
        self._max_delay = 60.0   # Cap at 1 minute
        self._factor = 2         # Double the wait each time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_consumer(self, stream_name: str, callback: Callable) -> None:
        """
        Spawn a long-running listener Task for *stream_name*.

        For every message that arrives a dedicated handler Task is created
        (see :meth:`_spawn_handler`).  A no-op if a live listener for that
        stream already exists.
        """
        existing = self._listeners.get(stream_name)
        if existing and not existing.done():
            log.warning(f"[March Hare] Listener for '{stream_name}' is already running.")
            return

        # Allow restart after stop() was called
        self._stop_event.clear()

        task = asyncio.create_task(
            self._listener_loop(callback, stream_name),
            name=f"march_hare:listener:{stream_name}",
        )
        self._listeners[stream_name] = task
        task.add_done_callback(lambda t: self._on_listener_done(stream_name, t))

        log.debug(f"[March Hare] Listener task started for stream '{stream_name}'.")

    def mark_ready(self) -> None:
        """
        Signal that the lizard is fully bootstrapped.

        Until this is called, listener loops will read from the streams but
        will not spawn handler tasks—messages received during startup are
        silently skipped to avoid acting on events with a not-yet-ready system.
        """
        self._ready_event.set()
        log.debug("[March Hare] System ready — handlers will be spawned for incoming events.")

    async def stop(self) -> None:
        """Signal all listeners and pending handlers to stop and await clean termination."""
        self._stop_event.set()
        self._ready_event.clear()

        all_tasks = list(self._listeners.values()) + list(self._handlers.values())
        for task in all_tasks:
            task.cancel()

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        self._listeners.clear()
        self._handlers.clear()
        log.debug("[March Hare] All tasks stopped.")

    async def notify_event(self, event_type: str, payload: Dict, stream_name: str) -> None:
        """Publish an event to a Redis Stream."""
        try:
            event = {
                "event_type": event_type,
                "payload": json.dumps(payload),
                "source_pod": self.pod_id,
            }

            await self._async_redis_client.xadd(
                name=stream_name,
                fields=event,
                maxlen=STREAM_MAX_LEN,
                approximate=True,
            )

            log.debug(f"[March Hare] Event '{event_type}' sent to stream '{stream_name}'.")
        except Exception as e:
            log.error(f"[March Hare] Error publishing to Redis: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_handler(self, callback: Callable, stream_name: str, message_id: str, data: dict) -> None:
        """
        Create a short-lived asyncio Task to process a single message.

        The task is registered in :attr:`_handlers` on creation and removed
        automatically via a done-callback once it completes (successfully,
        with an error, or cancelled).
        """
        handler_id = f"{stream_name}:{message_id}"
        task = asyncio.create_task(
            callback(data),
            name=f"march_hare:handler:{handler_id}",
        )
        self._handlers[handler_id] = task
        task.add_done_callback(lambda t: self._on_handler_done(handler_id, t))

        log.debug(f"[March Hare] Handler spawned for message '{message_id}' on '{stream_name}'.")

    def _on_listener_done(self, stream_name: str, task: asyncio.Task) -> None:
        """Remove a finished listener from the registry and log the outcome."""
        self._listeners.pop(stream_name, None)

        if task.cancelled():
            log.debug(f"[March Hare] Listener for '{stream_name}' was cancelled.")
        elif (exc := task.exception()) is not None:
            log.error(f"[March Hare] Listener for '{stream_name}' raised an exception: {exc}")
        else:
            log.debug(f"[March Hare] Listener for '{stream_name}' completed.")

    def _on_handler_done(self, handler_id: str, task: asyncio.Task) -> None:
        """Remove a finished handler from the registry and log the outcome."""
        self._handlers.pop(handler_id, None)

        if task.cancelled():
            log.debug(f"[March Hare] Handler '{handler_id}' was cancelled.")
        elif (exc := task.exception()) is not None:
            log.error(f"[March Hare] Handler '{handler_id}' raised an exception: {exc}")
        else:
            log.debug(f"[March Hare] Handler '{handler_id}' completed successfully.")

    async def _listener_loop(self, callback: Callable, stream_name: str) -> None:
        """
        Continuously poll *stream_name* and spawn a handler Task per message.

        The loop starts reading from the stream immediately (so no events are
        missed due to startup timing) but defers handler spawning until the
        system signals readiness via :meth:`mark_ready`.

        Backoff state is local to each invocation so that concurrent listeners
        for different streams never interfere with each other.
        """
        last_id = "$"
        retries = 0

        log.debug(f"[March Hare] Listener loop started for stream '{stream_name}'.")

        while not self._stop_event.is_set():
            try:
                # block=100 ms so the stop flag is checked frequently
                events = await self._async_redis_client.xread(
                    {stream_name: last_id}, count=10, block=100
                )

                if retries > 0:
                    log.info(f"[March Hare] Redis reconnected for '{stream_name}'. Resetting backoff.")
                    retries = 0

                if not events:
                    continue

                # Do not spawn handlers until the lizard is fully ready
                if not self._ready_event.is_set():
                    # Advance the cursor so we don't replay startup-era messages
                    # once the system becomes ready
                    for _stream, messages in events:
                        for message_id, _data in messages:
                            last_id = message_id
                    continue

                for _stream, messages in events:
                    for message_id, data in messages:
                        self._spawn_handler(callback, stream_name, message_id, data)
                        # Advance cursor immediately; each message is handled by
                        # its own independent Task
                        last_id = message_id

            except asyncio.CancelledError:
                log.debug(f"[March Hare] Listener loop for '{stream_name}' cancelled.")
                raise  # let asyncio handle the cancellation cleanly

            except (aioredis.ConnectionError, aioredis.TimeoutError, ValueError) as e:
                if self._stop_event.is_set():
                    break

                jitter = random.uniform(0, 1)
                delay = min(self._max_delay, (self._base_delay * (self._factor ** retries)) + jitter)

                log.warning(
                    f"[March Hare] Redis unavailable on '{stream_name}': {e}. "
                    f"Retrying in {delay:.2f}s (attempt {retries + 1})."
                )

                await asyncio.sleep(delay)
                retries += 1

            except Exception as e:
                if self._stop_event.is_set():
                    break

                log.error(f"[March Hare] Unexpected error in listener loop for '{stream_name}': {e}")
                await asyncio.sleep(2)  # brief pause before retrying non-connection errors


_march_hare: Optional[MarchHare] = None


async def _handle_plugin_event(lizard, data: dict) -> None:
    """Dispatch a plugin event received from the Redis Stream."""
    try:
        event_type = data.get("event_type")
        source_pod = data.get("source_pod")
        payload = json.loads(data.get("payload", "{}"))

        if source_pod == _march_hare.pod_id:
            log.debug(f"[March Hare] Ignoring self-originated event: '{event_type}'.")
            return

        if event_type == MarchHareConfig.events["PLUGIN_INSTALLATION"]:
            # the instance that installed the plugin already stored the active plugins
            await lizard.plugin_manager.install_extracted_plugin(payload["plugin_id"], persist=False)
            lizard.activate_plugin_endpoints(payload["plugin_id"])
            return

        if event_type == MarchHareConfig.events["PLUGIN_SYSTEM_TOGGLE"]:
            # the event only says that something changed: the state is read from the database, so duplicated,
            # late or out-of-order events cannot leave this POD behind
            await lizard.sync_system_plugins()
            return

        if event_type == MarchHareConfig.events["PLUGIN_UNINSTALLATION"]:
            await lizard.uninstall_plugin(payload["plugin_id"], False)
            return

        log.warning(f"[March Hare] Unknown event type: '{event_type}'.")
    except Exception as e:
        log.error(f"[March Hare] Error processing plugin event: {e}")


# ------------------------------------------------------------------
# Lifecycle hooks
# ------------------------------------------------------------------

@hook(priority=0)
async def before_lizard_bootstrap(lizard) -> None:
    """
    Create the MarchHare singleton and start stream listeners early.

    Listeners begin reading from streams here so that no events published
    during the bootstrap window are missed.  Handler dispatching is gated
    by :meth:`MarchHare.mark_ready` and will only start once the lizard
    is fully initialised (see :func:`after_lizard_bootstrap`).
    """
    global _march_hare
    _march_hare = MarchHare()

    _march_hare.start_consumer(  # type: ignore[union-attr]
        stream_name=MarchHareConfig.streams["PLUGIN_EVENTS"],
        callback=lambda data: _handle_plugin_event(lizard, data),
    )


@hook(priority=0)
async def after_lizard_bootstrap(lizard) -> None:
    """
    Unlock handler dispatching now that the lizard is fully initialised.
    """
    global _march_hare

    if _march_hare is None:
        message = (
            "For some reason, the March Hare plugin is inactive. This could be due to a Redis connection "
            "failure during initialization. Please check the logs for more details. "
            "The plugin is fundamental to manage the PODs. The system cannot be used without it."
        )
        log.error(message)
        raise Exception(message)

    _march_hare.mark_ready()

    # the events published while this POD was starting were not handled (the listener reads only the new ones, and
    # dispatches nothing until now): align once with the system plugins stored in the database
    try:
        await lizard.sync_system_plugins()
    except Exception as e:
        log.error(f"[March Hare] Could not align the system plugins with the database: {e}")


@hook(priority=0)
async def before_lizard_shutdown(lizard) -> None:
    global _march_hare

    if _march_hare is not None:
        await _march_hare.stop()
        _march_hare = None


# ------------------------------------------------------------------
# Notification hooks
# ------------------------------------------------------------------

@hook(priority=999)
async def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard) -> None:
    global _march_hare

    if plugin_id and lizard.plugin_manager.plugins.get(plugin_id):
        await _march_hare.notify_event(  # type: ignore[union-attr]
            event_type=MarchHareConfig.events["PLUGIN_INSTALLATION"],
            payload={
                "plugin_id": plugin_id,
                "plugin_path": plugin_path,
            },
            stream_name=MarchHareConfig.streams["PLUGIN_EVENTS"],
        )


@hook(priority=0)
async def lizard_notify_plugin_uninstallation(plugin_id, lizard) -> None:
    global _march_hare

    if lizard.plugin_manager.plugins.get(plugin_id) is None:
        await _march_hare.notify_event(  # type: ignore[union-attr]
            event_type=MarchHareConfig.events["PLUGIN_UNINSTALLATION"],
            payload={"plugin_id": plugin_id},
            stream_name=MarchHareConfig.streams["PLUGIN_EVENTS"],
        )


@hook(priority=0)
async def after_plugin_toggling_on_system(plugin_id: str, lizard) -> None:
    global _march_hare

    await _march_hare.notify_event(  # type: ignore[union-attr]
        event_type=MarchHareConfig.events["PLUGIN_SYSTEM_TOGGLE"],
        payload={"plugin_id": plugin_id},
        stream_name=MarchHareConfig.streams["PLUGIN_EVENTS"],
    )
