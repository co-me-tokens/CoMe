import rich
import pytest
import inspect
import functools
import typing as T
from abc import ABC, abstractmethod
from contextlib import contextmanager



T_Decorated = T.TypeVar("T_Decorated")
T_Callable  = T.TypeVar("T_Callable", bound=T.Callable)


class FeatureNotAvailableError(pytest.skip.Exception):
    """An exception class that tells user current implementation is not usable due to feature shortage."""
    ...


def report_feature_status(description: bool = False):
    """Print a formatted report of all registered features and their availability status."""
    from rich.table import Table
    from rich import box
    
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Feature Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="left")
    
    if description:
        table.add_column("Description", style="dim", vertical="top")
    
    for feature_cls in FeatureSet.all_feats:
        if feature_cls.available:
            status = "[green]✓[/green]"
        else:
            status = "[red]✗[/red]"
        
        if description:
            table.add_row(f"[bold]{feature_cls.name}[/bold]", status, feature_cls.description)
        else:
            table.add_row(f"[bold]{feature_cls.name}[/bold]", status)
    
    rich.print(table)


class FeatureSet(ABC):
    available: T.ClassVar[bool] = False
    feat_name: T.ClassVar[str]
    description: T.ClassVar[str]
    
    all_feats: T.ClassVar[set[type["FeatureSet"]]] = set()
    
    def __init_subclass__(cls, **kwargs) -> None:
        cls.name       = kwargs.get("feat_name", cls.__name__)
        cls.description= kwargs.get("description", "(No Description)")
        cls.available  = cls.is_available()
        
        FeatureSet.all_feats.add(cls)
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @T.final
    @classmethod
    def register(cls):
        def decorator(func: T_Decorated) -> T_Decorated:
            if cls.available:
                return func
            elif inspect.isclass(func):
                class _PlaceHolderErrorMeta(type):
                    def __getattribute__(self, name: str):
                        # Allow access to special methods and some essential attributes
                        if name in ('__name__', '__qualname__', '__module__', '__class__', 
                                    '__init__', '__new__', '__getattribute__'):
                            return super().__getattribute__(name)
                        raise FeatureNotAvailableError(
                            f"Feature set {cls.name} is not available. "
                            f"The class {func.__name__} cannot be used."
                        )
                
                class _PlaceHolderErrorClass(metaclass=_PlaceHolderErrorMeta):
                    def __init__(self, *args, **kwargs) -> None:
                        raise FeatureNotAvailableError(f"Feature set {cls.name} is not available. The class {func.__name__} cannot be used.")
                    def __getattribute__(self, name: str):
                        raise FeatureNotAvailableError(f"Feature set {cls.name} is not available. The class {func.__name__} cannot be used.")
                
                _PlaceHolderErrorClass.__name__     = func.__name__
                _PlaceHolderErrorClass.__qualname__ = getattr(func, "__qualname__", func.__name__)
                                
                return _PlaceHolderErrorClass   # type: ignore # This is a monkey patch to replace the class with this 'Not Available Class'
            elif callable(func):
                @functools.wraps(func)          # type: ignore # This is a monkey patch to replace the function with this dummy function.
                def _PlaceHoderErrorFunc(*args, **kwargs):
                    raise FeatureNotAvailableError(f"Feature set {cls.name} is not available. The function {func.__name__} cannot be used.")
                
                return _PlaceHoderErrorFunc     # type: ignore #
            else:
                return None                     # type: ignore
        
        return decorator


class CUDAExtension(
    FeatureSet,
    feat_name="CUDA Extension",
    description="Custom CUDA kernels for optimal performance."):
    _force_disabled: T.ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            from ..cuda_extension import co_me_cuext, flash_attn_varlen_qkvpacked_func
            return True
        except BaseException:
            # return False
            raise

    @classmethod
    @contextmanager
    def disable(cls) -> T.Generator[None, None, None]:
        """
        Context manager that forces all CUDAExtension-accelerated functions to use
        their PyTorch fallback within the enclosed scope.

        Example:
            with CUDAExtension.disable():
                loss = differentiable_pipeline(data)
        """
        prev = cls._force_disabled
        cls._force_disabled = True
        try: yield
        finally: cls._force_disabled = prev

    @classmethod
    def accelerate(cls, cuda_func_name: str) -> T.Callable[[T_Callable], T_Callable]:
        """
        Decorator to use cuda-accelerated implementation if available.
        
        The decorated function serves as the fallback (slow) implementation.
        If CUDAExtension is available, the cuda implementation is lazily imported
        from the specified module and used instead.  Dispatch is decided at
        call time so that CUDAExtension.disable() can override it within its scope.
        
        Args:
            cuda_func_name: Name of the accelerated function in cuda_extension modeul.
        
        Example:
            @CUDAExtension.accelerate("forward_cuda")
            def forward_slow(...):
                # PyTorch fallback implementation
                ...
        """
        def decorator(fallback_func: T_Callable) -> T_Callable:
            if not cls.available:
                return fallback_func

            from ..cuda_extension import co_me_cuext
            cuda_func = getattr(co_me_cuext, cuda_func_name)

            @functools.wraps(fallback_func)
            def _dispatch(*args, **kwargs):
                if CUDAExtension._force_disabled:
                    return fallback_func(*args, **kwargs)
                return cuda_func(*args, **kwargs)

            return _dispatch  # type: ignore
        return decorator


if __name__ == "__main__":
    report_feature_status(description=True)
