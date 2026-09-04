import json
import logging
import os
import sys
import time
import traceback
from abc import abstractmethod, ABC
from pprint import pformat
from typing import Callable
from loguru import logger

from cat.env import get_env


def get_log_level():
    """Return the global LOG level."""
    return get_env("CAT_LOG_LEVEL")


class CatLogEngine:
    """The log engine.

    Engine to filter the logs in the terminal according to the level of severity.

    Attributes
    ----------
    LOG_LEVEL: str
        Level of logging set in the `.env` file.
    _plugin_log_handlers: list
        A list of callable functions registered by plugins to handle log messages.

    Notes
    -----
    The logging level set in the `.env` file will print all the logs from that level to above.
    Available levels are:

        - `DEBUG`
        - `INFO`
        - `WARNING`
        - `ERROR`
        - `CRITICAL`

    Default to `CAT_LOG_LEVEL` env variable (`INFO`).
    """
    def __init__(self):
        self.LOG_LEVEL = get_log_level()
        self.default_log()
        self._plugin_log_handlers = []  # Initialize the list for plugin handlers

        # workaround for pdfminer logging
        # https://github.com/pdfminer/pdfminer.six/issues/347
        logging.getLogger("pdfminer").setLevel(logging.WARNING)

    def show_log_level(self, record: dict):
        """Allows to show stuff in the log based on the global setting.

        Args:
            record: dict

        Returns:
            bool
        """
        return record["level"].no >= logger.level(self.LOG_LEVEL).no

    def default_log(self):
        """Set the same debug level to all the project dependencies."""
        t = "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green>"
        level = "<level>{level}:</level>"
        # origin = "<level>{extra[original_name]}.{extra[original_class]}.{extra[original_caller]}::{extra[original_line]}</level>"
        message = "<level>{message}</level>"
        log_format = f"{t} {level}\t{message}"

        logger.remove()
        logger.add(  # type: ignore
            sys.stdout,
            level=self.LOG_LEVEL,
            colorize=True,
            format=log_format,
            # backtrace=True,
            # diagnose=True,
            filter=self.show_log_level,
        )

    def __call__(self, msg, level="DEBUG"):
        """Alias of self.log()"""
        self.log(msg, level)

    def _print_short_traceback(self):
        """Print a short traceback of the last exception."""
        from cat.utils import colored_text

        if sys.exc_info()[0] is not None:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for err in formatted_traceback:
                print(colored_text(err, "red"))

    def debug(self, msg):
        """Logs a DEBUG message"""
        self.log(msg, level="DEBUG")

    def info(self, msg):
        """Logs an INFO message"""
        self.log(msg, level="INFO")

    def warning(self, msg):
        """Logs a WARNING message"""
        self.log(msg, level="WARNING")

    def error(self, msg):
        """Logs an ERROR message"""
        self.log(msg, level="ERROR")
        self._print_short_traceback()

    def critical(self, msg):
        """Logs a CRITICAL message"""
        self.log(msg, level="CRITICAL")
        self._print_short_traceback()

    def log(self, msg, level="DEBUG"):
        """Log a message and dispatch to registered plugin handlers.

        Args:
            msg: Message to be logged.
            level (str): Logging level."""
        # prettify
        if isinstance(msg, str):
            pass
        elif type(msg) in [dict, list]:  # TODO: should be recursive
            try:
                msg = json.dumps(msg, indent=4)
            except TypeError: # Catch potential TypeError during serialization
                msg = pformat(msg) # Fallback to pformat if JSON serialization fails
        else:
            msg = pformat(msg)

        # actual log to stdout using loguru
        lines = msg.split("\n")
        for line in lines:
            logger.log(level, line)

        # Dispatch to plugin handlers
        for handler in self._plugin_log_handlers:
            try:
                handler(msg, level)
            except Exception as e:
                # Log any errors in plugin handlers so they don't break the main logging
                logger.error(f"Error in plugin log handler: {e}", exc_info=True)

    def register_plugin_log_handler(self, handler_func: Callable):
        """Registers a function from a plugin to receive log messages.

        The `handler_func` should accept two arguments: `msg` (str) and `level` (str).
        """
        if handler_func in self._plugin_log_handlers:
            self.warning(f"Attempted to register a log handler that is already registered: {handler_func.__name__}")
            return

        if callable(handler_func):
            self._plugin_log_handlers.append(handler_func)
            self.info(f"Registered plugin log handler: {handler_func.__name__}")
            return

        self.warning(f"Attempted to register non-callable as log handler: {handler_func}")

    def unregister_plugin_log_handler(self, handler_func: Callable):
        """Unregisters a previously registered plugin log handler."""
        if handler_func in self._plugin_log_handlers:
            self._plugin_log_handlers.remove(handler_func)
            self.info(f"Unregistered plugin log handler: {handler_func.__name__}")

            return

        self.warning(f"Attempted to unregister a log handler that was not registered: {handler_func}")

    def welcome(self):
        from cat.utils import get_base_path, get_base_url

        # Atomically claim the one-shot welcome banner. `O_CREAT | O_EXCL` means
        # only the first process (e.g. the first uvicorn worker) prints the
        # banner and creates the flag; concurrent workers hit FileExistsError
        # and return. The check-then-write version raced under `--workers N`.
        try:
            fd = os.open(".welcome", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return

        try:
            cat_docs_address = os.path.join(get_base_url().strip("/"), "docs")
            print("\n\n")
            try:
                with open(get_base_path() + "welcome.txt", "r") as f:
                    print(f.read())
                    time.sleep(0.01)
            except FileNotFoundError:
                self.warning("welcome.txt not found. Skipping welcome message.")

            print("\n=============== ^._.^ ===============\n")
            print(f"Cat REST API:   {cat_docs_address}")
            print("======================================")
        finally:
            os.close(fd)  # keep the flag file (content is irrelevant)

    def log_examples(self):
        """Log examples for the log engine."""
        for c in [self, "Hello there!", {"ready", "set", "go"}, [1, 4, "sdfsf"], {"a": 1, "b": {"c": 2}}]:
            self.debug(c)
            self.info(c)
            self.warning(c)
            self.error(c)
            self.critical(c)

        def intentional_error():
            _ = 42/0
        try:
            intentional_error()
        except ZeroDivisionError: # Catch specific error for clarity
            self.error("This error is just for demonstration purposes.")


class CatLogProcessor(ABC):
    def __init__(self):
        self.log_engine = log # Get reference to the running log engine

        # IMMEDIATELY REGISTER THE HANDLER upon plugin instantiation
        self.log_engine.register_plugin_log_handler(self.handle_log_message)
        self.log_engine.info(f"{self.__class__.__name__}: registered its log handler.")

    @abstractmethod
    def handle_log_message(self, message: str, level: str):
        """
        This method will be called by the CatLogEngine's `log` method for every log message.
        Implement this method to process the log message as needed by your plugin.

        Args:
            message (str): The log message.
            level (str): The log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        """
        pass

    # IMPORTANT: Implement a method for plugin shutdown/cleanup
    # The Cheshire Cat Core should ideally call this when unloading the plugin.
    def __del__(self):
        """
        Called when the plugin is being unloaded. Essential for cleanup.
        """
        self.log_engine.unregister_plugin_log_handler(self.handle_log_message)
        self.log_engine.info(f"{self.__class__.__name__}: unregistered its log handler.")


# logger instance
log = CatLogEngine()
