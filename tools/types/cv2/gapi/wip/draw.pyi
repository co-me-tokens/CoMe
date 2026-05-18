from __future__ import annotations
__all__: list[str] = ['Circle', 'Image', 'Line', 'Mosaic', 'Poly', 'Rect', 'Text', 'render', 'render3ch', 'renderNV12']
class Circle:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Image:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Line:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Mosaic:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Poly:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Rect:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Text:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def render(*args, **kwargs) -> None:
    """
    .   @brief The function renders on the input image passed drawing primitivies
    .   
    .   @param bgr input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @param prims vector of drawing primitivies
    .   @param args graph compile time parameters
    
    
    
    render(y_plane, uv_plane, prims[, args]) -> None
    .   @brief The function renders on two NV12 planes passed drawing primitivies
    .   
    .   @param y_plane input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @param uv_plane input image: 8-bit unsigned 2-channel image @ref CV_8UC2.
    .   @param prims vector of drawing primitivies
    .   @param args graph compile time parameters
    """
def render3ch(src, prims) -> retval:
    """
    .   @brief Renders on 3 channels input
    .   
    .   Output image must be 8-bit unsigned planar 3-channel image
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3
    .   @param prims draw primitives
    """
def renderNV12(y, uv, prims) -> retval:
    """
    .   @brief Renders on two planes
    .   
    .   Output y image must be 8-bit unsigned planar 1-channel image @ref CV_8UC1
    .   uv image must be 8-bit unsigned planar 2-channel image @ref CV_8UC2
    .   
    .   @param y  input image: 8-bit unsigned 1-channel image @ref CV_8UC1
    .   @param uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2
    .   @param prims draw primitives
    """
