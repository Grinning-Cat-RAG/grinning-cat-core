import pickle
from datetime import datetime, timedelta, timezone
from typing import List
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.util import datetime_to_utc_timestamp
from pydantic import BaseModel, Field

from cat import utils
from cat.log import log
from cat.db.database import get_sync_db
from cat.utils import singleton


class JobStatus(utils.Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    REMOVED = "removed"


class Job(BaseModel):
    id: str
    name: str
    next_run: datetime | None = None
    status: JobStatus = Field(default=JobStatus.SCHEDULED)


class _SharedRedisJobStore(RedisJobStore):
    """The job store shared by every node. Every node processes the same due jobs, so the writes after a run are
    idempotent: a job already removed (by the node that ran it, or on purpose) is not updated back to life, and
    removing it again is not an error (otherwise the scheduler of the node stops, never to run a job again)."""

    # update a job only while it exists, atomically: KEYS jobs, run_times; ARGV id, state, next run timestamp or ""
    _UPDATE_IF_EXISTS = """
    if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 0 then return 0 end
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
    if ARGV[3] == '' then redis.call('ZREM', KEYS[2], ARGV[1]) else redis.call('ZADD', KEYS[2], ARGV[3], ARGV[1]) end
    return 1
    """

    def update_job(self, job):
        next_run = datetime_to_utc_timestamp(job.next_run_time) if job.next_run_time else ""
        self.redis.eval(
            self._UPDATE_IF_EXISTS, 2, self.jobs_key, self.run_times_key,
            job.id, pickle.dumps(job.__getstate__(), self.pickle_protocol), next_run,
        )

    def remove_job(self, job_id):
        with self.redis.pipeline() as pipe:
            pipe.hdel(self.jobs_key, job_id)
            pipe.zrem(self.run_times_key, job_id)
            pipe.execute()


class _OncePerRunExecutor(AsyncIOExecutor):
    """Every node reads the due jobs from the shared store and submits them: a run goes to the node that books its
    scheduled time first (a key per job and time, expiring by itself: nothing to release, a dead node blocks nothing).
    The other nodes skip it silently."""

    #: How long a booking lasts: well beyond the time a late node may still submit the same run
    BOOKING_SECONDS = 3600

    def __init__(self, client_db, on_submitted):
        super().__init__()
        self._client_db = client_db
        self._on_submitted = on_submitted

    def submit_job(self, job, run_times):
        booked = [
            run_time for run_time in run_times
            if self._client_db.set(
                f"white_rabbit:run:{job.id}:{run_time.timestamp()}", "booked", nx=True, ex=self.BOOKING_SECONDS,
            )
        ]
        if not booked:
            return  # another node runs them
        super().submit_job(job, booked)
        self._on_submitted(job.id)


@singleton
class WhiteRabbit:
    """
    The WhiteRabbit

    Here the cron magic happens. This class is responsible for scheduling and executing jobs. It uses APScheduler under
    the hood, but we can easily swap it for another library if needed. The main idea is to have a centralized place to
    manage all the scheduled tasks in the system, from sending a message to re-embedding MCP tools periodically.
    """
    def __init__(self):
        log.debug("Initializing WhiteRabbit...")

        self._client_db = get_sync_db()
        self._prefix_lock_key = "white_rabbit:lock"
        self._prefix_status_key = "white_rabbit:status"

        # Get connection kwargs from the existing client, but create a SEPARATE raw (decode_responses=False) connection
        # exclusively for APScheduler's RedisJobStore, which stores pickled binary data and cannot use a UTF-8 decoding
        # client. This avoids any interference with the main client connection used by the rest of the system.
        jobstores = {
            "default": _SharedRedisJobStore(
                jobs_key="white_rabbit:jobs",
                run_times_key="white_rabbit:run_times",
                **(self._client_db.get_connection_kwargs() | {"decode_responses": False}),
            )
        }

        executors = {
            "default": _OncePerRunExecutor(self._client_db, self._job_submitted),
            "processpool": ProcessPoolExecutor(5),
        }

        # the runs missed while no node was up (a crash, a restart) are merged into one: a single late run (within the
        # grace time, otherwise skipped and logged once), not one per missed run
        job_defaults = {"coalesce": True, "max_instances": 10}

        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone.utc,
        )

        # Listen to execution and error events to track status accurately (the submission is tracked by the executor:
        # only the node running the job sets it running)
        self.scheduler.add_listener(self._job_ended_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

        self.jobs: List[str] = []

        log.debug("WhiteRabbit: Starting scheduler")

        try:
            self.scheduler.start()
            log.debug("WhiteRabbit: Scheduler started")
            self._is_running = True
        except Exception as e:
            log.error(f"WhiteRabbit: Error during scheduler start: {e}")
            self._is_running = False

    def __del__(self):
        self.shutdown()

    def _status_key(self, job_id: str) -> str:
        return f"{self._prefix_status_key}:{job_id}"

    def _set_job_status(self, job_id: str, status: JobStatus):
        if status in [JobStatus.COMPLETED, JobStatus.REMOVED]:
            # No need to retain completed or removed status; clean up
            self._client_db.delete(self._status_key(job_id))
        else:
            self._client_db.set(self._status_key(job_id), str(status))

    def _get_job_status(self, job_id: str) -> JobStatus:
        raw = self._client_db.get(self._status_key(job_id))
        if raw is None:
            return JobStatus.SCHEDULED  # Default for jobs with no explicit status set yet
        return JobStatus(raw.decode() if isinstance(raw, bytes) else raw)

    def _job_submitted(self, job_id: str):
        """Triggered when this node submits a job for execution."""
        self._set_job_status(job_id, JobStatus.RUNNING)
        log.debug(f"WhiteRabbit: job {job_id} is now running.")

    def _job_ended_listener(self, event):
        """Triggered when a job ends (success or error)."""
        if event.exception:
            log.error(
                f"WhiteRabbit: error during the execution of job {event.job_id} "
                f"started at {event.scheduled_run_time}. Error: {event.traceback}"
            )
        else:
            log.info(
                f"WhiteRabbit: executed job {event.job_id} started at {event.scheduled_run_time}. "
                f"Value returned: {event.retval}"
            )

        self._set_job_status(event.job_id, JobStatus.COMPLETED)

        # Remove one-shot jobs from our tracking list after completion
        if event.job_id in self.jobs:
            job = self.scheduler.get_job(event.job_id)
            if job is None:
                self.jobs.remove(event.job_id)

    def acquire_lock(self, event: str) -> bool:
        lock_key = f"{self._prefix_lock_key}:{event}"
        lock_acquired = self._client_db.set(lock_key, "locked", nx=True, ex=3600)

        debug_message = (
            f"WhiteRabbit: Lock acquired. Starting '{event}' event..."
            if lock_acquired
            else f"WhiteRabbit: Job '{event}' already in execution or handled by another node."
        )
        log.debug(debug_message)

        return lock_acquired  # type: ignore[no-any-return]

    def release_lock(self, event: str):
        lock_key = f"{self._prefix_lock_key}:{event}"
        self._client_db.delete(lock_key)
        log.debug(f"WhiteRabbit: Lock released for '{event}' event.")  # type: ignore[attr-defined]

    def shutdown(self):
        """Stop the scheduler of this node. The jobs stay in the store, shared by every node: they belong to the
        instance, not to the node that scheduled them (another node, or this one at the restart, runs them). A job goes
        only when removed on purpose (``remove_job``)."""
        if self._is_running:
            log.info("WhiteRabbit: Scheduler stopped")
            self.scheduler.shutdown(wait=True)
            self._is_running = False

        try:
            self._client_db.close()
        except Exception:
            pass

    def get_job(self, job_id: str) -> Job | None:
        """
        Gets a scheduled job.

        Args:
            job_id (str): The id assigned to the job.

        Returns:
            Job | None: A Job object if the job exists, otherwise None.
        """
        job = self.scheduler.get_job(job_id)
        if not job:
            return None

        return Job(
            id=job.id,
            name=job.name,
            next_run=job.next_run_time,
            status=self._get_job_status(job.id),
        )

    def get_jobs(self) -> List[Job]:
        """
        Returns a list of scheduled jobs.

        Returns:
            List[Job]: A list of Job objects.
        """
        return [
            Job(
                id=job.id,
                name=job.name,
                next_run=job.next_run_time,
                status=self._get_job_status(job.id),
            )
            for job in self.scheduler.get_jobs()
        ]

    def pause_job(self, job_id: str) -> bool:
        """
        Pauses a scheduled job.

        Args:
            job_id (str): The id assigned to the job.

        Returns:
            bool: The outcome of the pause action.
        """
        try:
            self.scheduler.pause_job(job_id)
            log.info(f"WhiteRabbit: paused job {job_id}")  # type: ignore[attr-defined]
            return True
        except Exception as e:
            log.error(f"WhiteRabbit: error during job pause. {e}")  # type: ignore[attr-defined]
            return False

    def resume_job(self, job_id: str) -> bool:
        """
        Resumes a paused job.

        Args:
            job_id (str): The id assigned to the job.

        Returns:
            bool: The outcome of the resume action.
        """
        try:
            self.scheduler.resume_job(job_id)
            log.info(f"WhiteRabbit: resumed job {job_id}")  # type: ignore[attr-defined]
            return True
        except Exception as e:
            log.error(f"WhiteRabbit: error during job resume. {e}")  # type: ignore[attr-defined]
            return False

    def remove_job(self, job_id: str) -> bool:
        """
        Removes a scheduled job.

        Args:
            job_id (str): The id assigned to the job.

        Returns:
            bool: The outcome of the removal.
        """
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                self.jobs.remove(job_id)
            self._set_job_status(job_id, JobStatus.REMOVED)
            log.info(f"WhiteRabbit: Removed job {job_id}")  # type: ignore[attr-defined]
            return True  # also when already gone (removed by another node, or never existed): the store allows it
        except Exception as e:
            log.error(f"WhiteRabbit: error during job removal. {e}")  # type: ignore[attr-defined]
            return False

    def schedule_job(
        self,
        job,
        job_id: str = None,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
        microseconds: int = 0,
        **kwargs,
    ) -> str:
        """
        Schedule a one-shot job. It stays in the store shared by every node until it runs or is removed, also when the
        node that scheduled it stops: a fixed ``job_id`` already pending (e.g. scheduled again at every startup) raises
        ``ConflictingIdError``, so check it with ``get_job`` first.

        Args:
            job (function): The function to be called.
            job_id (str): The id assigned to the job.
            days (int | str): Days to wait.
            hours (int | str): Hours to wait.
            minutes (int | str): Minutes to wait.
            seconds (int | str): Seconds to wait.
            milliseconds (int | str): Milliseconds to wait.
            microseconds (int | str): Microseconds to wait.
            **kwargs: The arguments to pass to the function.

        Returns:
            The job id.
        """
        schedule = datetime.today() + timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            milliseconds=milliseconds,
            microseconds=microseconds,
        )

        # Check that the function is callable
        if not callable(job):
            log.error("WhiteRabbit: The job should be callable!")
            raise TypeError(f"TypeError: '{type(job)}' object is not callable")

        # Generate id if none
        if job_id is None:
            job_id = f"{job.__name__}-{schedule.strftime('%m/%d/%Y-%H:%M:%S')}"

        # Schedule the job
        self.scheduler.add_job(job, "date", id=job_id, run_date=schedule, kwargs=kwargs)
        self.jobs.append(job_id)

        log.info(f"WhiteRabbit: Scheduled job '{job_id}' to run at {schedule}")
        return job_id

    def schedule_interval_job(
        self,
        job,
        job_id: str = None,  # type: ignore[assignment]
        start_date: datetime = None,  # type: ignore[assignment]
        end_date: datetime = None,  # type: ignore[assignment]
        days: int | str = 0,
        hours: int | str = 0,
        minutes: int | str = 0,
        seconds: int | str = 0,
        **kwargs,
    ) -> str:
        """
        Schedule an interval job.

        Args:
            job (function): The function to be called.
            job_id (str): The id assigned to the job.
            start_date (datetime): Start date. If None the job starts instantaneously.
            end_date (datetime): End date. If None the job never ends.
            days (int | str): Days interval.
            hours (int | str): Hours interval.
            minutes (int | str): Minutes interval.
            seconds (int | str): Seconds interval.
            **kwargs: The arguments to pass to the function.

        Returns:
            The job id.
        """
        # Check that the function is callable
        if not callable(job):
            log.error("WhiteRabbit: The job should be callable!")  # type: ignore[attr-defined]
            raise TypeError(f"TypeError: '{type(job)}' object is not callable")

        # Generate id if none
        if job_id is None:
            job_id = f"{job.__name__}-interval-{days}-{hours}-{minutes}-{seconds}"

        # Schedule the job
        # replace_existing=True makes scheduling idempotent across workers/replicas:
        # they all share the same RedisJobStore, so without it a second worker adding
        # the same job id raises ConflictingIdError (and get→remove→add is not atomic).
        self.scheduler.add_job(
            job,
            "interval",
            id=job_id,
            start_date=start_date,
            end_date=end_date,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            kwargs=kwargs,
            replace_existing=True,
        )
        self.jobs.append(job_id)

        log.info(f"WhiteRabbit: Scheduled interval job '{job_id}' with interval {days}d {hours}h {minutes}m {seconds}s")
        return job_id

    def schedule_cron_job(
        self,
        job,
        job_id: str = None,  # type: ignore[assignment]
        start_date: datetime = None,  # type: ignore[assignment]
        end_date: datetime = None,  # type: ignore[assignment]
        year: int | str = None,
        month: int | str = None,
        day: int | str = None,
        week: int | str =None,
        day_of_week: int | str = None,
        hour: int | str = None,
        minute: int | str = None,
        second: int | str = None,
        **kwargs,
    ) -> str:
        """
        Schedule a cron job.

        Args:
            job (function): The function to be called.
            job_id (str): The id assigned to the job.
            start_date (datetime): Start date. If None the job starts instantaneously.
            end_date (datetime): End date. If None the job never ends.
            year (int | str): 4-digit year
            month (int | str): month (1-12)
            day (int | str): day of month (1-31)
            week (int | str): ISO week (1-53)
            day_of_week (int | str): number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)
            hour (int | str): hour (0-23)
            minute (int | str): minute (0-59)
            second (int | str): second (0-59)
            **kwargs: The arguments to pass to the function.

        Returns:
            The job id.
        """
        # Check that the function is callable
        if not callable(job):
            log.error("WhiteRabbit: The job should be callable!")  # type: ignore[attr-defined]
            raise TypeError(f"TypeError: '{type(job)}' object is not callable")

        # Generate id if none
        if job_id is None:
            job_id = f"{job.__name__}-cron"

        # Schedule the job
        self.scheduler.add_job(
            job,
            "cron",
            id=job_id,
            start_date=start_date,
            end_date=end_date,
            year=year,
            month=month,
            day=day,
            week=week,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            second=second,
            kwargs=kwargs,
            replace_existing=True,  # idempotent across nodes and restarts: the store is shared and outlives them
        )
        self.jobs.append(job_id)

        log.info(f"WhiteRabbit: Scheduled cron job '{job_id}' with cron params year={year} month={month} day={day} week={week} day_of_week={day_of_week} hour={hour} minute={minute} second={second}")
        return job_id

    def schedule_chat_message(
        self,
        content: str,
        cat,
        days: float = 0,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0,
        milliseconds: float = 0,
        microseconds: float = 0,
    ) -> str:
        """
        Schedule a chat message.

        Args:
            content (str): The message to be sent.
            cat (StrayCat): Stray Cat instance.
            days (int | str): Days to wait.
            hours (int | str): Hours to wait.
            minutes (int | str): Minutes to wait.
            seconds (int | str): Seconds to wait.
            milliseconds (int | str): Milliseconds to wait.
            microseconds (int | str): Microseconds to wait.

        Returns:
            The job id.
        """
        # Calculate time
        schedule = datetime.today() + timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            milliseconds=milliseconds,
            microseconds=microseconds,
        )

        # Generate id
        job_id = f"send_ws_message-{schedule.strftime('%m/%d/%Y-%H:%M:%S')}"

        # Schedule the job
        self.scheduler.add_job(
            cat.notifier.send_chat_message,
            trigger="date",
            id=job_id,
            run_date=schedule,
            kwargs={"message": content},
        )
        self.jobs.append(job_id)

        log.info(f"WhiteRabbit: Scheduled chat message '{job_id}' to be sent at {schedule}")
        return job_id
