import logging
import typing as tp
from contextlib import contextmanager
from pathlib import Path
from rich.logging import RichHandler


class Diagnostics:
    __is_activate: tp.ClassVar[bool] = False
    __logger: tp.ClassVar[logging.Logger | None] = None
    __file_handler: tp.ClassVar[logging.FileHandler | None] = None

    @classmethod
    def is_active(cls) -> bool: return cls.__is_activate

    @classmethod
    @contextmanager
    def activate(cls, is_active: bool=True) -> tp.Iterator[None]:
        current_state = cls.__is_activate
        
        try:
            cls.__is_activate = is_active
            yield
        finally:
            cls.__is_activate = current_state

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        if cls.__logger is not None:
            return cls.__logger

        logger = logging.getLogger("Diagnostic")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = RichHandler(
                show_time=False,
                show_level=False,
                show_path=False,
                markup=False,
            )
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [Diagnostic] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            logger.addHandler(handler)

        cls.__logger = logger
        return logger

    @classmethod
    def set_file_output(cls, path: Path) -> None:
        cls.clear_file_output()
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        cls._get_logger().addHandler(fh)
        cls.__file_handler = fh

    @classmethod
    def clear_file_output(cls) -> None:
        if cls.__file_handler is not None:
            cls._get_logger().removeHandler(cls.__file_handler)
            cls.__file_handler.close()
            cls.__file_handler = None

    @classmethod
    def log(cls, message: str) -> None:
        if not cls.__is_activate:
            return

        cls._get_logger().info(message, stacklevel=2)
