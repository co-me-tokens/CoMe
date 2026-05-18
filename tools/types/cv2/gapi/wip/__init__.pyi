from __future__ import annotations
from cv2.gapi.wip.gst import GStreamerPipeline
from . import draw
from . import gst
from . import onevpl
__all__: list[str] = ['GOutputs', 'GStreamerPipeline', 'IStreamSource', 'draw', 'get_streaming_source', 'gst', 'make_capture_src', 'make_gst_src', 'onevpl']
class GOutputs:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def getGArray(type) -> retval:
        """
        .
        """
    @staticmethod
    def getGMat() -> retval:
        """
        .
        """
    @staticmethod
    def getGOpaque(type) -> retval:
        """
        .
        """
    @staticmethod
    def getGScalar() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class IStreamSource:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def get_streaming_source(*args, **kwargs) -> retval:
    """
    .
    """
def make_capture_src(path) -> retval:
    """
    .   * @brief OpenCV's VideoCapture-based streaming source.
    .    *
    .    * This class implements IStreamSource interface.
    .    * Its constructor takes the same parameters as cv::VideoCapture does.
    .    *
    .    * Please make sure that videoio OpenCV module is available before using
    .    * this in your application (G-API doesn't depend on it directly).
    .    *
    .    * @note stream sources are passed to G-API via shared pointers, so
    .    *  please gapi::make_src<> to create objects and ptr() to pass a
    .    *  GCaptureSource to cv::gin().
    
    
    
    make_capture_src(id) -> retval
    .
    """
def make_gst_src(*args, **kwargs) -> retval:
    """
    .
    """
