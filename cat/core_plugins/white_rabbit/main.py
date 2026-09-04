from datetime import UTC, datetime

import cat.db.cruds.settings as crud_settings
from cat import BillTheLizard, hook, log
from cat.core_plugins.white_rabbit.white_rabbit import WhiteRabbit

scheduled_job_id = "re_embed_tools"


# IMPORTANT: This function MUST live at a module level (not inside another function) so that APScheduler + Redis can
# pickle/serialize it by its fully qualified import path.
# All runtime context is passed explicitly via kwargs.
async def re_embed_tools():
    """Re-embed all procedures (tools, forms, MCP clients) for all CheshireCat instances"""
    global scheduled_job_id

    lizard = BillTheLizard()

    lock_acquired = lizard.white_rabbit.acquire_lock(scheduled_job_id)
    if not lock_acquired:
        return
    try:
        ccat_ids = await crud_settings.get_agents_main_keys()
        # Track errors to ensure we don't leave things hanging
        for ccat_id in ccat_ids:
            if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
                continue
            try:
                await ccat.embed_procedures()
                del ccat
            except Exception as e:
                log.error(f"WhiteRabbit: Failed re-embedding for Cat {ccat_id}: {e}")
    finally:
        # release the lock immediately
        lizard.white_rabbit.release_lock(scheduled_job_id)


@hook
async def after_lizard_bootstrap(lizard: BillTheLizard):
    # Start scheduling system and attach it to the BillTheLizard core class
    lizard.white_rabbit = WhiteRabbit()

    try:
        settings = await lizard.plugin_manager.get_plugin().load_settings()
        interval_job_days = int(settings["embed_procedures_every_n_days"])
    except (ValueError, KeyError):
        interval_job_days = None

    if not interval_job_days:
        return

    # schedule_interval_job uses replace_existing=True, so re-scheduling the same
    # job id is idempotent across the N workers that run this hook against the shared
    # RedisJobStore. No explicit get/remove needed (and get→remove→add is racy).
    # start_date=now makes the first fire happen right after bootstrap (restart),
    # then every `days` relative to that anchor.
    lizard.white_rabbit.schedule_interval_job(
        job=re_embed_tools,
        job_id=scheduled_job_id,
        days=interval_job_days,
        start_date=datetime.now(UTC),
    )


@hook(priority=0)
def before_lizard_shutdown(lizard) -> None:
    lizard.white_rabbit.shutdown()
    lizard.white_rabbit = None


@hook(priority=0)
def after_plugin_toggling_on_system(plugin_id: str, lizard: BillTheLizard) -> None:
    global scheduled_job_id

    this_plugin = lizard.plugin_manager.get_plugin()
    if plugin_id != this_plugin.id:
        return

    active_plugins = lizard.plugin_manager.active_plugins
    if plugin_id not in active_plugins and lizard.white_rabbit.get_job(scheduled_job_id):
        lizard.white_rabbit.remove_job(scheduled_job_id)
