from __future__ import annotations
__all__: list[str] = ['setParallelForBackend']
def setParallelForBackend(*args, **kwargs) -> retval:
    """
    .   @brief Change OpenCV parallel_for backend
    .    *
    .    * @note This call is not thread-safe. Consider calling this function from the `main()` before any other OpenCV processing functions (and without any other created threads).
    """
