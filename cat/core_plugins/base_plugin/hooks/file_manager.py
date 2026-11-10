"""Hook to handle operations after file-manager deletions."""

from cat import hook


@hook(priority=0)
def before_file_manager_file_delete(filename: str, scope: str, cat) -> None:
    """
    Hook triggered before a stored file's memory points are deleted.

    Fired by the DELETE /file_manager/... single-file route right after the
    source file is removed from storage and BEFORE the memory points are
    deleted (their metadata still records the extracted-image file names).
    No-op default.
    """


@hook(priority=0)
def after_file_manager_file_deleted(filename: str, scope: str, cat) -> None:
    """
    Hook triggered after a stored file (and its memory points) is deleted.

    Fired by the DELETE /file_manager/... routes after the file and its points
    are removed. No-op default.
    """
