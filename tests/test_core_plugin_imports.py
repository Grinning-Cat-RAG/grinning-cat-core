def test_the_white_rabbit_logs_with_the_logger_of_the_cat():
    # regression: white_rabbit is imported while the `cat` package initializes (by cat.mixins), when `from cat import
    # log` still finds the module `cat.log`, not the logger: every log call of the scheduler failed
    from cat.core_plugins.white_rabbit import white_rabbit
    from cat.log import CatLogEngine

    assert isinstance(white_rabbit.log, CatLogEngine)
