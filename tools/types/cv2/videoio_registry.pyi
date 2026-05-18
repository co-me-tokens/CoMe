from __future__ import annotations
__all__: list[str] = ['getBackendName', 'getBackends', 'getCameraBackendPluginVersion', 'getCameraBackends', 'getStreamBackendPluginVersion', 'getStreamBackends', 'getWriterBackendPluginVersion', 'getWriterBackends', 'hasBackend', 'isBackendBuiltIn']
def getBackendName(api) -> retval:
    """
    .   @brief Returns backend API name or "UnknownVideoAPI(xxx)"
    .   @param api backend ID (#VideoCaptureAPIs)
    """
def getBackends() -> retval:
    """
    .   @brief Returns list of all available backends
    """
def getCameraBackendPluginVersion(api) -> retval, version_ABI, version_API:
    """
    .   @brief Returns description and ABI/API version of videoio plugin's camera interface
    """
def getCameraBackends() -> retval:
    """
    .   @brief Returns list of available backends which works via `cv::VideoCapture(int index)`
    """
def getStreamBackendPluginVersion(api) -> retval, version_ABI, version_API:
    """
    .   @brief Returns description and ABI/API version of videoio plugin's stream capture interface
    """
def getStreamBackends() -> retval:
    """
    .   @brief Returns list of available backends which works via `cv::VideoCapture(filename)`
    """
def getWriterBackendPluginVersion(api) -> retval, version_ABI, version_API:
    """
    .   @brief Returns description and ABI/API version of videoio plugin's writer interface
    """
def getWriterBackends() -> retval:
    """
    .   @brief Returns list of available backends which works via `cv::VideoWriter()`
    """
def hasBackend(api) -> retval:
    """
    .   @brief Returns true if backend is available
    """
def isBackendBuiltIn(api) -> retval:
    """
    .   @brief Returns true if backend is built in (false if backend is used as plugin)
    """
