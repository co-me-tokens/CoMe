from __future__ import annotations
import cv2 as cv
from cv2 import gapi as _native
import sys as sys
from . import core
from . import ie
from . import oak
from . import onnx
from . import own
from . import render
from . import streaming
from . import video
from . import wip
__all__: list = ['op', 'kernel']
class GArray:
    class Any:
        def __new__(self):
            ...
    class Bool:
        def __new__(self):
            ...
    class Double:
        def __new__(self):
            ...
    class Float:
        def __new__(self):
            ...
    class GMat:
        def __new__(self):
            ...
    class Int:
        def __new__(self):
            ...
    class Mat:
        def __new__(self):
            ...
    class Point:
        def __new__(self):
            ...
    class Point2f:
        def __new__(self):
            ...
    class Point3f:
        def __new__(self):
            ...
    class Prim:
        def __new__(self):
            ...
    class Rect:
        def __new__(self):
            ...
    class Scalar:
        def __new__(self):
            ...
    class Size:
        def __new__(self):
            ...
    class String:
        def __new__(self):
            ...
    @classmethod
    def __new__(cls, argtype):
        ...
class GNetPackage:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GNetParam:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GOpaque:
    class Any:
        def __new__(self):
            ...
    class Bool:
        def __new__(self):
            ...
    class Double:
        def __new__(self):
            ...
    class Float:
        def __new__(self):
            ...
    class Int:
        def __new__(self):
            ...
    class Point:
        def __new__(self):
            ...
    class Point2f:
        def __new__(self):
            ...
    class Point3f:
        def __new__(self):
            ...
    class Prim:
        def __new__(self):
            ...
    class Rect:
        def __new__(self):
            ...
    class Size:
        def __new__(self):
            ...
    class String:
        def __new__(self):
            ...
    @classmethod
    def __new__(cls, argtype):
        ...
def BGR2Gray(src) -> retval:
    """
    .   @brief Converts an image from BGR color space to gray-scaled.
    .   
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   Resulting gray color value computed as
    .   \\f[\\texttt{dst} (I)= \\texttt{0.114} * \\texttt{src}(I).B + \\texttt{0.587} * \\texttt{src}(I).G  + \\texttt{0.299} * \\texttt{src}(I).R \\f]
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2gray"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
    .   @sa BGR2LUV
    """
def BGR2I420(src) -> retval:
    """
    .   @brief Converts an image from BGR color space to I420 color space.
    .   
    .   The function converts an input image from BGR color space to I420.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 1-channel image. @ref CV_8UC1.
    .   Width of I420 output image must be the same as width of input image.
    .   Height of I420 output image must be equal 3/2 from height of input image.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2i420"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa I4202BGR
    """
def BGR2LUV(src) -> retval:
    """
    .   @brief Converts an image from BGR color space to LUV color space.
    .   
    .   The function converts an input image from BGR color space to LUV.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2luv"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa RGB2Lab, RGB2LUV
    """
def BGR2RGB(src) -> retval:
    """
    .   @brief Converts an image from BGR color space to RGB color space.
    .   
    .   The function converts an input image from BGR color space to RGB.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image is 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2rgb"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa RGB2BGR
    """
def BGR2YUV(src) -> retval:
    """
    .   @brief Converts an image from BGR color space to YUV color space.
    .   
    .   The function converts an input image from BGR color space to YUV.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2yuv"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa YUV2BGR
    """
def BayerGR2RGB(src_gr) -> retval:
    """
    .   @brief Converts an image from BayerGR color space to RGB.
    .   The function converts an input image from BayerGR color space to RGB.
    .   The conventional ranges for G, R, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.bayergr2rgb"
    .   
    .   @param src_gr input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   
    .   @sa YUV2BGR, NV12toRGB
    """
def Canny(*args, **kwargs) -> retval:
    """
    .   @brief Finds edges in an image using the Canny algorithm.
    .   
    .   The function finds edges in the input image and marks them in the output map edges using the
    .   Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
    .   largest value is used to find initial segments of strong edges. See
    .   <http://en.wikipedia.org/wiki/Canny_edge_detector>
    .   
    .   @note Function textual ID is "org.opencv.imgproc.feature.canny"
    .   
    .   @param image 8-bit input image.
    .   @param threshold1 first threshold for the hysteresis procedure.
    .   @param threshold2 second threshold for the hysteresis procedure.
    .   @param apertureSize aperture size for the Sobel operator.
    .   @param L2gradient a flag, indicating whether a more accurate \\f$L_2\\f$ norm
    .   \\f$=\\sqrt{(dI/dx)^2 + (dI/dy)^2}\\f$ should be used to calculate the image gradient magnitude (
    .   L2gradient=true ), or whether the default \\f$L_1\\f$ norm \\f$=|dI/dx|+|dI/dy|\\f$ is enough (
    .   L2gradient=false ).
    """
def GIn(*args):
    ...
def GOut(*args):
    ...
def I4202BGR(src) -> retval:
    """
    .   @brief Converts an image from I420 color space to BGR color space.
    .   
    .   The function converts an input image from I420 color space to BGR.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image. @ref CV_8UC3.
    .   Width of BGR output image must be the same as width of input image.
    .   Height of BGR output image must be equal 2/3 from height of input image.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.i4202bgr"
    .   
    .   @param src input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @sa BGR2I420
    """
def I4202RGB(src) -> retval:
    """
    .   @brief Converts an image from I420 color space to BGR color space.
    .   
    .   The function converts an input image from I420 color space to BGR.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image. @ref CV_8UC3.
    .   Width of RGB output image must be the same as width of input image.
    .   Height of RGB output image must be equal 2/3 from height of input image.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.i4202rgb"
    .   
    .   @param src input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @sa RGB2I420
    """
def LUT(src, lut) -> retval:
    """
    .   @brief Performs a look-up table transform of a matrix.
    .   
    .   The function LUT fills the output matrix with values from the look-up table. Indices of the entries
    .   are taken from the input matrix. That is, the function processes each element of src as follows:
    .   \\f[\\texttt{dst} (I)  \\leftarrow \\texttt{lut(src(I))}\\f]
    .   
    .   Supported matrix data types are @ref CV_8UC1.
    .   Output is a matrix of the same size and number of channels as src, and the same depth as lut.
    .   
    .   @note Function textual ID is "org.opencv.core.transform.LUT"
    .   
    .   @param src input matrix of 8-bit elements.
    .   @param lut look-up table of 256 elements; in case of multi-channel input array, the table should
    .   either have a single channel (in this case the same table is used for all channels) or the same
    .   number of channels as in the input matrix.
    """
def LUV2BGR(src) -> retval:
    """
    .   @brief Converts an image from LUV color space to BGR color space.
    .   
    .   The function converts an input image from LUV color space to BGR.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.luv2bgr"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa BGR2LUV
    """
def Laplacian(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the Laplacian of an image.
    .   
    .   The function calculates the Laplacian of the source image by adding up the second x and y
    .   derivatives calculated using the Sobel operator:
    .   
    .   \\f[\\texttt{dst} =  \\Delta \\texttt{src} =  \\frac{\\partial^2 \\texttt{src}}{\\partial x^2} +  \\frac{\\partial^2 \\texttt{src}}{\\partial y^2}\\f]
    .   
    .   This is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image
    .   with the following \\f$3 \\times 3\\f$ aperture:
    .   
    .   \\f[\\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\\f]
    .   
    .   @note Function textual ID is "org.opencv.imgproc.filters.laplacian"
    .   
    .   @param src Source image.
    .   @param ddepth Desired depth of the destination image.
    .   @param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
    .   details. The size must be positive and odd.
    .   @param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
    .   applied. See #getDerivKernels for details.
    .   @param delta Optional delta value that is added to the results prior to storing them in dst .
    .   @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
    .   @return Destination image of the same size and the same number of channels as src.
    .   @sa  Sobel, Scharr
    """
def NV12toBGR(src_y, src_uv) -> retval:
    """
    .   @brief Converts an image from NV12 (YUV420p) color space to BGR.
    .   The function converts an input image from NV12 color space to RGB.
    .   The conventional ranges for Y, U, and V channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.nv12tobgr"
    .   
    .   @param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.
    .   
    .   @sa YUV2BGR, NV12toRGB
    """
def NV12toGray(src_y, src_uv) -> retval:
    """
    .   @brief Converts an image from NV12 (YUV420p) color space to gray-scaled.
    .   The function converts an input image from NV12 color space to gray-scaled.
    .   The conventional ranges for Y, U, and V channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.nv12togray"
    .   
    .   @param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.
    .   
    .   @sa YUV2RGB, NV12toBGR
    """
def NV12toRGB(src_y, src_uv) -> retval:
    """
    .   @brief Converts an image from NV12 (YUV420p) color space to RGB.
    .   The function converts an input image from NV12 color space to RGB.
    .   The conventional ranges for Y, U, and V channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.nv12torgb"
    .   
    .   @param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
    .   @param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.
    .   
    .   @sa YUV2RGB, NV12toBGR
    """
def RGB2Gray(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to gray-scaled.
    .   
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   Resulting gray color value computed as
    .   \\f[\\texttt{dst} (I)= \\texttt{0.299} * \\texttt{src}(I).R + \\texttt{0.587} * \\texttt{src}(I).G  + \\texttt{0.114} * \\texttt{src}(I).B \\f]
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2gray"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
    .   @sa RGB2YUV
    
    
    
    RGB2Gray(src, rY, gY, bY) -> retval
    .   @overload
    .   Resulting gray color value computed as
    .   \\f[\\texttt{dst} (I)= \\texttt{rY} * \\texttt{src}(I).R + \\texttt{gY} * \\texttt{src}(I).G  + \\texttt{bY} * \\texttt{src}(I).B \\f]
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2graycustom"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
    .   @param rY float multiplier for R channel.
    .   @param gY float multiplier for G channel.
    .   @param bY float multiplier for B channel.
    .   @sa RGB2YUV
    """
def RGB2HSV(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to HSV.
    .   The function converts an input image from RGB color space to HSV.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2hsv"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @sa YUV2BGR, NV12toRGB
    """
def RGB2I420(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to I420 color space.
    .   
    .   The function converts an input image from RGB color space to I420.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 1-channel image. @ref CV_8UC1.
    .   Width of I420 output image must be the same as width of input image.
    .   Height of I420 output image must be equal 3/2 from height of input image.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2i420"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa I4202RGB
    """
def RGB2Lab(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to Lab color space.
    .   
    .   The function converts an input image from BGR color space to Lab.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC1.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2lab"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
    .   @sa RGB2YUV, RGB2LUV
    """
def RGB2YUV(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to YUV color space.
    .   
    .   The function converts an input image from RGB color space to YUV.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   In case of linear transformations, the range does not matter. But in case of a non-linear
    .   transformation, an input RGB image should be normalized to the proper value range to get the correct
    .   results, like here, at RGB \\f$\\rightarrow\\f$ Y\\*u\\*v\\* transformation.
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2yuv"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa YUV2RGB, RGB2Lab
    """
def RGB2YUV422(src) -> retval:
    """
    .   @brief Converts an image from RGB color space to YUV422.
    .   The function converts an input image from RGB color space to YUV422.
    .   The conventional ranges for R, G, and B channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 2-channel image @ref CV_8UC2.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2yuv422"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @sa YUV2BGR, NV12toRGB
    """
def Sobel(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
    .   
    .   In all cases except one, the \\f$\\texttt{ksize} \\times \\texttt{ksize}\\f$ separable kernel is used to
    .   calculate the derivative. When \\f$\\texttt{ksize = 1}\\f$, the \\f$3 \\times 1\\f$ or \\f$1 \\times 3\\f$
    .   kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
    .   or the second x- or y- derivatives.
    .   
    .   There is also the special value `ksize = FILTER_SCHARR (-1)` that corresponds to the \\f$3\\times3\\f$ Scharr
    .   filter that may give more accurate results than the \\f$3\\times3\\f$ Sobel. The Scharr aperture is
    .   
    .   \\f[\\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\\f]
    .   
    .   for the x-derivative, or transposed for the y-derivative.
    .   
    .   The function calculates an image derivative by convolving the image with the appropriate kernel:
    .   
    .   \\f[\\texttt{dst} =  \\frac{\\partial^{xorder+yorder} \\texttt{src}}{\\partial x^{xorder} \\partial y^{yorder}}\\f]
    .   
    .   The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
    .   resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
    .   or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
    .   case corresponds to a kernel of:
    .   
    .   \\f[\\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\\f]
    .   
    .   The second case corresponds to a kernel of:
    .   
    .   \\f[\\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\\f]
    .   
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.sobel"
    .   
    .   @param src input image.
    .   @param ddepth output image depth, see @ref filter_depths "combinations"; in the case of
    .       8-bit input images it will result in truncated derivatives.
    .   @param dx order of the derivative x.
    .   @param dy order of the derivative y.
    .   @param ksize size of the extended Sobel kernel; it must be odd.
    .   @param scale optional scale factor for the computed derivative values; by default, no scaling is
    .   applied (see cv::getDerivKernels for details).
    .   @param delta optional delta value that is added to the results prior to storing them in dst.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa filter2D, gaussianBlur, cartToPolar
    """
def SobelXY(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
    .   
    .   In all cases except one, the \\f$\\texttt{ksize} \\times \\texttt{ksize}\\f$ separable kernel is used to
    .   calculate the derivative. When \\f$\\texttt{ksize = 1}\\f$, the \\f$3 \\times 1\\f$ or \\f$1 \\times 3\\f$
    .   kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
    .   or the second x- or y- derivatives.
    .   
    .   There is also the special value `ksize = FILTER_SCHARR (-1)` that corresponds to the \\f$3\\times3\\f$ Scharr
    .   filter that may give more accurate results than the \\f$3\\times3\\f$ Sobel. The Scharr aperture is
    .   
    .   \\f[\\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\\f]
    .   
    .   for the x-derivative, or transposed for the y-derivative.
    .   
    .   The function calculates an image derivative by convolving the image with the appropriate kernel:
    .   
    .   \\f[\\texttt{dst} =  \\frac{\\partial^{xorder+yorder} \\texttt{src}}{\\partial x^{xorder} \\partial y^{yorder}}\\f]
    .   
    .   The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
    .   resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
    .   or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
    .   case corresponds to a kernel of:
    .   
    .   \\f[\\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\\f]
    .   
    .   The second case corresponds to a kernel of:
    .   
    .   \\f[\\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\\f]
    .   
    .   @note
    .    - First returned matrix correspons to dx derivative while the second one to dy.
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.sobelxy"
    .   
    .   @param src input image.
    .   @param ddepth output image depth, see @ref filter_depths "combinations"; in the case of
    .       8-bit input images it will result in truncated derivatives.
    .   @param order order of the derivatives.
    .   @param ksize size of the extended Sobel kernel; it must be odd.
    .   @param scale optional scale factor for the computed derivative values; by default, no scaling is
    .   applied (see cv::getDerivKernels for details).
    .   @param delta optional delta value that is added to the results prior to storing them in dst.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa filter2D, gaussianBlur, cartToPolar
    """
def YUV2BGR(src) -> retval:
    """
    .   @brief Converts an image from YUV color space to BGR color space.
    .   
    .   The function converts an input image from YUV color space to BGR.
    .   The conventional ranges for B, G, and R channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2bgr"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   @sa BGR2YUV
    """
def YUV2RGB(src) -> retval:
    """
    .   @brief Converts an image from YUV color space to RGB.
    .   The function converts an input image from YUV color space to RGB.
    .   The conventional ranges for Y, U, and V channel values are 0 to 255.
    .   
    .   Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2rgb"
    .   
    .   @param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
    .   
    .   @sa RGB2Lab, RGB2YUV
    """
def __op(*args, **kwargs) -> retval:
    ...
def absDiff(src1, src2) -> retval:
    """
    .   @brief Calculates the per-element absolute difference between two matrices.
    .   
    .   The function absDiff calculates absolute difference between two matrices of the same size and depth:
    .       \\f[\\texttt{dst}(I) =  \\texttt{saturate} (| \\texttt{src1}(I) -  \\texttt{src2}(I)|)\\f]
    .       where I is a multi-dimensional index of matrix elements. In case of
    .       multi-channel matrices, each channel is processed independently.
    .   Output matrix must have the same size and depth as input matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.absdiff"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    .   @sa abs
    """
def absDiffC(src, c) -> retval:
    """
    .   @brief Calculates absolute value of matrix elements.
    .   
    .   The function abs calculates absolute difference between matrix elements and given scalar value:
    .       \\f[\\texttt{dst}(I) =  \\texttt{saturate} (| \\texttt{src1}(I) -  \\texttt{matC}(I)|)\\f]
    .       where matC is constructed from given scalar c and has the same sizes and depth as input matrix src.
    .   
    .   Output matrix must be of the same size and depth as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.absdiffC"
    .   @param src input matrix.
    .   @param c scalar to be subtracted.
    .   @sa min, max
    """
def add(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element sum of two matrices.
    .   
    .   The function add calculates sum of two matrices of the same size and the same number of channels:
    .   \\f[\\texttt{dst}(I) =  \\texttt{saturate} ( \\texttt{src1}(I) +  \\texttt{src2}(I)) \\quad \\texttt{if mask}(I) \\ne0\\f]
    .   
    .   The function can be replaced with matrix expressions:
    .       \\f[\\texttt{dst} =  \\texttt{src1} + \\texttt{src2}\\f]
    .   
    .   The input matrices and the output matrix can all have the same or different depths. For example, you
    .   can add a 16-bit unsigned matrix to a 8-bit signed matrix and store the sum as a 32-bit
    .   floating-point matrix. Depth of the output matrix is determined by the ddepth parameter.
    .   If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
    .   the same depth as the input matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.add"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    .   @param ddepth optional depth of the output matrix.
    .   @sa sub, addWeighted
    """
def addC(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element sum of matrix and given scalar.
    .   
    .   The function addC adds a given scalar value to each element of given matrix.
    .   The function can be replaced with matrix expressions:
    .   
    .       \\f[\\texttt{dst} =  \\texttt{src1} + \\texttt{c}\\f]
    .   
    .   Depth of the output matrix is determined by the ddepth parameter.
    .   If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
    .   The matrices can be single or multi channel. Output matrix must have the same size and number of channels as the input matrix.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.addC"
    .   @param src1 first input matrix.
    .   @param c scalar value to be added.
    .   @param ddepth optional depth of the output matrix.
    .   @sa sub, addWeighted
    
    
    
    addC(c, src1[, ddepth]) -> retval
    .
    """
def addWeighted(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the weighted sum of two matrices.
    .   
    .   The function addWeighted calculates the weighted sum of two matrices as follows:
    .   \\f[\\texttt{dst} (I)= \\texttt{saturate} ( \\texttt{src1} (I)* \\texttt{alpha} +  \\texttt{src2} (I)* \\texttt{beta} +  \\texttt{gamma} )\\f]
    .   where I is a multi-dimensional index of array elements. In case of multi-channel matrices, each
    .   channel is processed independently.
    .   
    .   The function can be replaced with a matrix expression:
    .       \\f[\\texttt{dst}(I) =  \\texttt{alpha} * \\texttt{src1}(I) - \\texttt{beta} * \\texttt{src2}(I) + \\texttt{gamma} \\f]
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.addweighted"
    .   @param src1 first input matrix.
    .   @param alpha weight of the first matrix elements.
    .   @param src2 second input matrix of the same size and channel number as src1.
    .   @param beta weight of the second matrix elements.
    .   @param gamma scalar added to each sum.
    .   @param ddepth optional depth of the output matrix.
    .   @sa  add, sub
    """
def bilateralFilter(*args, **kwargs) -> retval:
    """
    .   @brief Applies the bilateral filter to an image.
    .   
    .   The function applies bilateral filtering to the input image, as described in
    .   http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    .   bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is
    .   very slow compared to most filters.
    .   
    .   _Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (\\<
    .   10), the filter will not have much effect, whereas if they are large (\\> 150), they will have a very
    .   strong effect, making the image look "cartoonish".
    .   
    .   _Filter size_: Large filters (d \\> 5) are very slow, so it is recommended to use d=5 for real-time
    .   applications, and perhaps d=9 for offline applications that need heavy noise filtering.
    .   
    .   This filter does not work inplace.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.filters.bilateralfilter"
    .   
    .   @param src Source 8-bit or floating-point, 1-channel or 3-channel image.
    .   @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
    .   it is computed from sigmaSpace.
    .   @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
    .   farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting
    .   in larger areas of semi-equal color.
    .   @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
    .   farther pixels will influence each other as long as their colors are close enough (see sigmaColor
    .   ). When d\\>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is
    .   proportional to sigmaSpace.
    .   @param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes
    .   @return Destination image of the same size and type as src.
    """
def bitwise_and(src1, src2) -> retval:
    """
    .   @brief computes bitwise conjunction of the two matrixes (src1 & src2)
    .   Calculates the per-element bit-wise logical conjunction of two matrices of the same size.
    .   
    .   In case of floating-point matrices, their machine-specific bit
    .   representations (usually IEEE754-compliant) are used for the operation.
    .   In case of multi-channel matrices, each channel is processed
    .   independently. Output matrix must have the same size and depth as the input
    .   matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.bitwise_and"
    .   
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    """
def bitwise_not(src) -> retval:
    """
    .   @brief Inverts every bit of an array.
    .   
    .   The function bitwise_not calculates per-element bit-wise inversion of the input
    .   matrix:
    .   \\f[\\texttt{dst} (I) =  \\neg \\texttt{src} (I)\\f]
    .   
    .   In case of floating-point matrices, their machine-specific bit
    .   representations (usually IEEE754-compliant) are used for the operation.
    .   In case of multi-channel matrices, each channel is processed
    .   independently. Output matrix must have the same size and depth as the input
    .   matrix.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.bitwise_not"
    .   
    .   @param src input matrix.
    """
def bitwise_or(src1, src2) -> retval:
    """
    .   @brief computes bitwise disjunction of the two matrixes (src1 | src2)
    .   Calculates the per-element bit-wise logical disjunction of two matrices of the same size.
    .   
    .   In case of floating-point matrices, their machine-specific bit
    .   representations (usually IEEE754-compliant) are used for the operation.
    .   In case of multi-channel matrices, each channel is processed
    .   independently. Output matrix must have the same size and depth as the input
    .   matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.bitwise_or"
    .   
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    """
def bitwise_xor(src1, src2) -> retval:
    """
    .   @brief computes bitwise logical "exclusive or" of the two matrixes (src1 ^ src2)
    .   Calculates the per-element bit-wise logical "exclusive or" of two matrices of the same size.
    .   
    .   In case of floating-point matrices, their machine-specific bit
    .   representations (usually IEEE754-compliant) are used for the operation.
    .   In case of multi-channel matrices, each channel is processed
    .   independently. Output matrix must have the same size and depth as the input
    .   matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.bitwise_xor"
    .   
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    """
def blur(*args, **kwargs) -> retval:
    """
    .   @brief Blurs an image using the normalized box filter.
    .   
    .   The function smooths an image using the kernel:
    .   
    .   \\f[\\texttt{K} =  \\frac{1}{\\texttt{ksize.width*ksize.height}} \\begin{bmatrix} 1 & 1 & 1 &  \\cdots & 1 & 1  \\\\ 1 & 1 & 1 &  \\cdots & 1 & 1  \\\\ \\hdotsfor{6} \\\\ 1 & 1 & 1 &  \\cdots & 1 & 1  \\\\ \\end{bmatrix}\\f]
    .   
    .   The call `blur(src, ksize, anchor, borderType)` is equivalent to `boxFilter(src, src.type(), ksize, anchor,
    .   true, borderType)`.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.blur"
    .   
    .   @param src Source image.
    .   @param ksize blurring kernel size.
    .   @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
    .   center.
    .   @param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur
    """
def boundingRect(src) -> retval:
    """
    .   @brief Calculates the up-right bounding rectangle of a point set or non-zero pixels
    .   of gray-scale image.
    .   
    .   The function calculates and returns the minimal up-right bounding rectangle for the specified
    .   point set or non-zero pixels of gray-scale image.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.imgproc.shape.boundingRectMat"
    .    - In case of a 2D points' set given, Mat should be 2-dimensional, have a single row or column
    .   if there are 2 channels, or have 2 columns if there is a single channel. Mat should have either
    .   @ref CV_32S or @ref CV_32F depth
    .   
    .   @param src Input gray-scale image @ref CV_8UC1; or input set of @ref CV_32S or @ref CV_32F
    .   2D points stored in Mat.
    """
def boxFilter(*args, **kwargs) -> retval:
    """
    .   @brief Blurs an image using the box filter.
    .   
    .   The function smooths an image using the kernel:
    .   
    .   \\f[\\texttt{K} =  \\alpha \\begin{bmatrix} 1 & 1 & 1 &  \\cdots & 1 & 1  \\\\ 1 & 1 & 1 &  \\cdots & 1 & 1  \\\\ \\hdotsfor{6} \\\\ 1 & 1 & 1 &  \\cdots & 1 & 1 \\end{bmatrix}\\f]
    .   
    .   where
    .   
    .   \\f[\\alpha = \\begin{cases} \\frac{1}{\\texttt{ksize.width*ksize.height}} & \\texttt{when } \\texttt{normalize=true}  \\\\1 & \\texttt{otherwise} \\end{cases}\\f]
    .   
    .   Unnormalized box filter is useful for computing various integral characteristics over each pixel
    .   neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
    .   algorithms, and so on). If you need to compute pixel sums over variable-size windows, use cv::integral.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.boxfilter"
    .   
    .   @param src Source image.
    .   @param dtype the output image depth (-1 to set the input image data type).
    .   @param ksize blurring kernel size.
    .   @param anchor Anchor position within the kernel. The default value \\f$(-1,-1)\\f$ means that the anchor
    .   is at the kernel center.
    .   @param normalize flag, specifying whether the kernel is normalized by its area or not.
    .   @param borderType Pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa  sepFilter, gaussianBlur, medianBlur, integral
    """
def cartToPolar(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the magnitude and angle of 2D vectors.
    .   
    .   The function cartToPolar calculates either the magnitude, angle, or both
    .   for every 2D vector (x(I),y(I)):
    .   \\f[\\begin{array}{l} \\texttt{magnitude} (I)= \\sqrt{\\texttt{x}(I)^2+\\texttt{y}(I)^2} , \\\\ \\texttt{angle} (I)= \\texttt{atan2} ( \\texttt{y} (I), \\texttt{x} (I))[ \\cdot180 / \\pi ] \\end{array}\\f]
    .   
    .   The angles are calculated with accuracy about 0.3 degrees. For the point
    .   (0,0), the angle is set to 0.
    .   
    .   First output is a matrix of magnitudes of the same size and depth as input x.
    .   Second output is a matrix of angles that has the same size and depth as
    .   x; the angles are measured in radians (from 0 to 2\\*Pi) or in degrees (0 to 360 degrees).
    .   
    .   @note Function textual ID is "org.opencv.core.math.cartToPolar"
    .   
    .   @param x matrix of @ref CV_32FC1 x-coordinates.
    .   @param y array of @ref CV_32FC1 y-coordinates.
    .   @param angleInDegrees a flag, indicating whether the angles are measured
    .   in radians (which is by default), or in degrees.
    .   @sa polarToCart
    """
def cmpEQ(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are equal to elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  ==  \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .       \\f[\\texttt{dst} =   \\texttt{src1} == \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpEQ"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpNE
    """
def cmpGE(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are greater or equal compare to elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  >= \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .       \\f[\\texttt{dst} =   \\texttt{src1} >= \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpGE"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpLE, cmpGT, cmpLT
    """
def cmpGT(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are greater compare to elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  > \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .   \\f[\\texttt{dst} =   \\texttt{src1} > \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices/matrix.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpGT"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpLE, cmpGE, cmpLT
    """
def cmpLE(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are less or equal compare to elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  <=  \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .       \\f[\\texttt{dst} =   \\texttt{src1} <= \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLE"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpGT, cmpGE, cmpLT
    """
def cmpLT(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are less than elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  < \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .       \\f[\\texttt{dst} =   \\texttt{src1} < \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices/matrix.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLT"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpLE, cmpGE, cmpGT
    """
def cmpNE(src1, src2) -> retval:
    """
    .   @brief Performs the per-element comparison of two matrices checking if elements from first matrix are not equal to elements in second.
    .   
    .   The function compares elements of two matrices src1 and src2 of the same size:
    .       \\f[\\texttt{dst} (I) =  \\texttt{src1} (I)  !=  \\texttt{src2} (I)\\f]
    .   
    .   When the comparison result is true, the corresponding element of output
    .   array is set to 255. The comparison operations can be replaced with the
    .   equivalent matrix expressions:
    .       \\f[\\texttt{dst} =   \\texttt{src1} != \\texttt{src2}\\f]
    .   
    .   Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    .       the input matrices.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.compare.cmpNE"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix/scalar of the same depth as first input matrix.
    .   @sa min, max, threshold, cmpEQ
    """
def compile_args(*args):
    ...
def concatHor(src1, src2) -> retval:
    """
    .   @brief Applies horizontal concatenation to given matrices.
    .   
    .   The function horizontally concatenates two GMat matrices (with the same number of rows).
    .   @code{.cpp}
    .       GMat A = { 1, 4,
    .                  2, 5,
    .                  3, 6 };
    .       GMat B = { 7, 10,
    .                  8, 11,
    .                  9, 12 };
    .   
    .       GMat C = gapi::concatHor(A, B);
    .       //C:
    .       //[1, 4, 7, 10;
    .       // 2, 5, 8, 11;
    .       // 3, 6, 9, 12]
    .   @endcode
    .   Output matrix must the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.transform.concatHor"
    .   
    .   @param src1 first input matrix to be considered for horizontal concatenation.
    .   @param src2 second input matrix to be considered for horizontal concatenation.
    .   @sa concatVert
    
    
    
    concatHor(v) -> retval
    .   @overload
    .   The function horizontally concatenates given number of GMat matrices (with the same number of columns).
    .   Output matrix must the same number of columns and depth as the input matrices, and the sum of rows of input matrices.
    .   
    .   @param v vector of input matrices to be concatenated horizontally.
    """
def concatVert(src1, src2) -> retval:
    """
    .   @brief Applies vertical concatenation to given matrices.
    .   
    .   The function vertically concatenates two GMat matrices (with the same number of cols).
    .    @code{.cpp}
    .       GMat A = { 1, 7,
    .                  2, 8,
    .                  3, 9 };
    .       GMat B = { 4, 10,
    .                  5, 11,
    .                  6, 12 };
    .   
    .       GMat C = gapi::concatVert(A, B);
    .       //C:
    .       //[1, 7;
    .       // 2, 8;
    .       // 3, 9;
    .       // 4, 10;
    .       // 5, 11;
    .       // 6, 12]
    .    @endcode
    .   
    .   Output matrix must the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.imgproc.transform.concatVert"
    .   
    .   @param src1 first input matrix to be considered for vertical concatenation.
    .   @param src2 second input matrix to be considered for vertical concatenation.
    .   @sa concatHor
    
    
    
    concatVert(v) -> retval
    .   @overload
    .   The function vertically concatenates given number of GMat matrices (with the same number of columns).
    .   Output matrix must the same number of columns and depth as the input matrices, and the sum of rows of input matrices.
    .   
    .   @param v vector of input matrices to be concatenated vertically.
    """
def convertTo(*args, **kwargs) -> retval:
    """
    .   @brief Converts a matrix to another data depth with optional scaling.
    .   
    .   The method converts source pixel values to the target data depth. saturate_cast\\<\\> is applied at
    .   the end to avoid possible overflows:
    .   
    .   \\f[m(x,y) = saturate \\_ cast<rType>( \\alpha (*this)(x,y) +  \\beta )\\f]
    .   Output matrix must be of the same size as input one.
    .   
    .   @note Function textual ID is "org.opencv.core.transform.convertTo"
    .   @param src input matrix to be converted from.
    .   @param rdepth desired output matrix depth or, rather, the depth since the number of channels are the
    .   same as the input has; if rdepth is negative, the output matrix will have the same depth as the input.
    .   @param alpha optional scale factor.
    .   @param beta optional delta added to the scaled values.
    """
def copy(in_) -> retval:
    """
    .   @brief Makes a copy of the input image. Note that this copy may be not real
    .   (no actual data copied). Use this function to maintain graph contracts,
    .   e.g when graph's input needs to be passed directly to output, like in Streaming mode.
    .   
    .   @note Function textual ID is "org.opencv.streaming.copy"
    .   
    .   @param in Input image
    .   @return Copy of the input
    """
def countNonZero(src) -> retval:
    """
    .   @brief Counts non-zero array elements.
    .   
    .   The function returns the number of non-zero elements in src :
    .   \\f[\\sum _{I: \\; \\texttt{src} (I) \\ne0 } 1\\f]
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.countNonZero"
    .   @param src input single-channel matrix.
    .   @sa  mean, min, max
    """
def crop(src, rect) -> retval:
    """
    .   @brief Crops a 2D matrix.
    .   
    .   The function crops the matrix by given cv::Rect.
    .   
    .   Output matrix must be of the same depth as input one, size is specified by given rect size.
    .   
    .   @note Function textual ID is "org.opencv.core.transform.crop"
    .   
    .   @param src input matrix.
    .   @param rect a rect to crop a matrix to
    .   @sa resize
    """
def descr_of(*args):
    ...
def dilate(*args, **kwargs) -> retval:
    """
    .   @brief Dilates an image by using a specific structuring element.
    .   
    .   The function dilates the source image using the specified structuring element that determines the
    .   shape of a pixel neighborhood over which the maximum is taken:
    .   \\f[\\texttt{dst} (x,y) =  \\max _{(x',y'):  \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\f]
    .   
    .   Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.dilate"
    .   
    .   @param src input image.
    .   @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
    .   structuring element is used. Kernel can be created using getStructuringElement
    .   @param anchor position of the anchor within the element; default value (-1, -1) means that the
    .   anchor is at the element center.
    .   @param iterations number of times dilation is applied.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of a constant border
    .   @sa  erode, morphologyEx, getStructuringElement
    """
def dilate3x3(*args, **kwargs) -> retval:
    """
    .   @brief Dilates an image by using 3 by 3 rectangular structuring element.
    .   
    .   The function dilates the source image using the specified structuring element that determines the
    .   shape of a pixel neighborhood over which the maximum is taken:
    .   \\f[\\texttt{dst} (x,y) =  \\max _{(x',y'):  \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\f]
    .   
    .   Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.dilate"
    .   
    .   @param src input image.
    .   @param iterations number of times dilation is applied.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of a constant border
    .   @sa  dilate, erode3x3
    """
def div(*args, **kwargs) -> retval:
    """
    .   @brief Performs per-element division of two matrices.
    .   
    .   The function divides one matrix by another:
    .   \\f[\\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\\f]
    .   
    .   For integer types when src2(I) is zero, dst(I) will also be zero.
    .   Floating point case returns Inf/NaN (according to IEEE).
    .   
    .   Different channels of
    .   multi-channel matrices are processed independently.
    .   The matrices can be single or multi channel. Output matrix must have the same size and depth as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.div"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix of the same size and depth as src1.
    .   @param scale scalar factor.
    .   @param ddepth optional depth of the output matrix; you can only pass -1 when src1.depth() == src2.depth().
    .   @sa  mul, add, sub
    """
def divC(*args, **kwargs) -> retval:
    """
    .   @brief Divides matrix by scalar.
    .   
    .   The function divC divides each element of matrix src by given scalar value:
    .   
    .   \\f[\\texttt{dst(I) = saturate(src(I)*scale/divisor)}\\f]
    .   
    .   When divisor is zero, dst(I) will also be zero. Different channels of
    .   multi-channel matrices are processed independently.
    .   The matrices can be single or multi channel. Output matrix must have the same size and depth as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.divC"
    .   @param src input matrix.
    .   @param divisor number to be divided by.
    .   @param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
    .   @param scale scale factor.
    .   @sa add, sub, div, addWeighted
    """
def divRC(*args, **kwargs) -> retval:
    """
    .   @brief Divides scalar by matrix.
    .   
    .   The function divRC divides given scalar by each element of matrix src and keep the division result in new matrix of the same size and type as src:
    .   
    .   \\f[\\texttt{dst(I) = saturate(divident*scale/src(I))}\\f]
    .   
    .   When src(I) is zero, dst(I) will also be zero. Different channels of
    .   multi-channel matrices are processed independently.
    .   The matrices can be single or multi channel. Output matrix must have the same size and depth as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.divRC"
    .   @param src input matrix.
    .   @param divident number to be divided.
    .   @param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
    .   @param scale scale factor
    .   @sa add, sub, div, addWeighted
    """
def equalizeHist(src) -> retval:
    """
    .   @brief Equalizes the histogram of a grayscale image.
    .   
    .   //! @} gapi_feature
    .   
    .   The function equalizes the histogram of the input image using the following algorithm:
    .   
    .   - Calculate the histogram \\f$H\\f$ for src .
    .   - Normalize the histogram so that the sum of histogram bins is 255.
    .   - Compute the integral of the histogram:
    .   \\f[H'_i =  \\sum _{0  \\le j < i} H(j)\\f]
    .   - Transform the image using \\f$H'\\f$ as a look-up table: \\f$\\texttt{dst}(x,y) = H'(\\texttt{src}(x,y))\\f$
    .   
    .   The algorithm normalizes the brightness and increases the contrast of the image.
    .   @note
    .    - The returned image is of the same size and type as input.
    .    - Function textual ID is "org.opencv.imgproc.equalizeHist"
    .   
    .   @param src Source 8-bit single channel image.
    """
def erode(*args, **kwargs) -> retval:
    """
    .   @brief Erodes an image by using a specific structuring element.
    .   
    .   The function erodes the source image using the specified structuring element that determines the
    .   shape of a pixel neighborhood over which the minimum is taken:
    .   
    .   \\f[\\texttt{dst} (x,y) =  \\min _{(x',y'):  \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\f]
    .   
    .   Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.erode"
    .   
    .   @param src input image
    .   @param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
    .   structuring element is used. Kernel can be created using getStructuringElement.
    .   @param anchor position of the anchor within the element; default value (-1, -1) means that the
    .   anchor is at the element center.
    .   @param iterations number of times erosion is applied.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of a constant border
    .   @sa  dilate, morphologyEx
    """
def erode3x3(*args, **kwargs) -> retval:
    """
    .   @brief Erodes an image by using 3 by 3 rectangular structuring element.
    .   
    .   The function erodes the source image using the rectangular structuring element with rectangle center as an anchor.
    .   Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.erode"
    .   
    .   @param src input image
    .   @param iterations number of times erosion is applied.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of a constant border
    .   @sa  erode, dilate3x3
    """
def filter2D(*args, **kwargs) -> retval:
    """
    .   @brief Convolves an image with the kernel.
    .   
    .   The function applies an arbitrary linear filter to an image. When
    .   the aperture is partially outside the image, the function interpolates outlier pixel values
    .   according to the specified border mode.
    .   
    .   The function does actually compute correlation, not the convolution:
    .   
    .   \\f[\\texttt{dst} (x,y) =  \\sum _{ \\substack{0\\leq x' < \\texttt{kernel.cols}\\\\{0\\leq y' < \\texttt{kernel.rows}}}}  \\texttt{kernel} (x',y')* \\texttt{src} (x+x'- \\texttt{anchor.x} ,y+y'- \\texttt{anchor.y} )\\f]
    .   
    .   That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
    .   the kernel using flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
    .   anchor.y - 1)`.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   Output image must have the same size and number of channels an input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.filter2D"
    .   
    .   @param src input image.
    .   @param ddepth desired depth of the destination image
    .   @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
    .   matrix; if you want to apply different kernels to different channels, split the image into
    .   separate color planes using split and process them individually.
    .   @param anchor anchor of the kernel that indicates the relative position of a filtered point within
    .   the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
    .   is at the kernel center.
    .   @param delta optional value added to the filtered pixels before storing them in dst.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa  sepFilter
    """
def flip(src, flipCode) -> retval:
    """
    .   @brief Flips a 2D matrix around vertical, horizontal, or both axes.
    .   
    .   The function flips the matrix in one of three different ways (row
    .   and column indices are 0-based):
    .   \\f[\\texttt{dst} _{ij} =
    .   \\left\\{
    .   \\begin{array}{l l}
    .   \\texttt{src} _{\\texttt{src.rows}-i-1,j} & if\\;  \\texttt{flipCode} = 0 \\\\
    .   \\texttt{src} _{i, \\texttt{src.cols} -j-1} & if\\;  \\texttt{flipCode} > 0 \\\\
    .   \\texttt{src} _{ \\texttt{src.rows} -i-1, \\texttt{src.cols} -j-1} & if\\; \\texttt{flipCode} < 0 \\\\
    .   \\end{array}
    .   \\right.\\f]
    .   The example scenarios of using the function are the following:
    .   *   Vertical flipping of the image (flipCode == 0) to switch between
    .       top-left and bottom-left image origin. This is a typical operation
    .       in video processing on Microsoft Windows\\* OS.
    .   *   Horizontal flipping of the image with the subsequent horizontal
    .       shift and absolute difference calculation to check for a
    .       vertical-axis symmetry (flipCode \\> 0).
    .   *   Simultaneous horizontal and vertical flipping of the image with
    .       the subsequent shift and absolute difference calculation to check
    .       for a central symmetry (flipCode \\< 0).
    .   *   Reversing the order of point arrays (flipCode \\> 0 or
    .       flipCode == 0).
    .   Output image must be of the same depth as input one, size should be correct for given flipCode.
    .   
    .   @note Function textual ID is "org.opencv.core.transform.flip"
    .   
    .   @param src input matrix.
    .   @param flipCode a flag to specify how to flip the array; 0 means
    .   flipping around the x-axis and positive value (for example, 1) means
    .   flipping around y-axis. Negative value (for example, -1) means flipping
    .   around both axes.
    .   @sa remap
    """
def gaussianBlur(*args, **kwargs) -> retval:
    """
    .   @brief Blurs an image using a Gaussian filter.
    .   
    .   The function filter2Ds the source image with the specified Gaussian kernel.
    .   Output image must have the same type and number of channels an input image.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .    - Function textual ID is "org.opencv.imgproc.filters.gaussianBlur"
    .   
    .   @param src input image;
    .   @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
    .   positive and odd. Or, they can be zero's and then they are computed from sigma.
    .   @param sigmaX Gaussian kernel standard deviation in X direction.
    .   @param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
    .   equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
    .   respectively (see cv::getGaussianKernel for details); to fully control the result regardless of
    .   possible future modifications of all this semantics, it is recommended to specify all of ksize,
    .   sigmaX, and sigmaY.
    .   @param borderType pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa  sepFilter, boxFilter, medianBlur
    """
def gin(*args):
    ...
def goodFeaturesToTrack(*args, **kwargs) -> retval:
    """
    .   @brief Determines strong corners on an image.
    .   
    .   The function finds the most prominent corners in the image or in the specified image region, as
    .   described in @cite Shi94
    .   
    .   -   Function calculates the corner quality measure at every source image pixel using the
    .       #cornerMinEigenVal or #cornerHarris .
    .   -   Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
    .       retained).
    .   -   The corners with the minimal eigenvalue less than
    .       \\f$\\texttt{qualityLevel} \\cdot \\max_{x,y} qualityMeasureMap(x,y)\\f$ are rejected.
    .   -   The remaining corners are sorted by the quality measure in the descending order.
    .   -   Function throws away each corner for which there is a stronger corner at a distance less than
    .       maxDistance.
    .   
    .   The function can be used to initialize a point-based tracker of an object.
    .   
    .   @note
    .    - If the function is called with different values A and B of the parameter qualityLevel , and
    .   A \\> B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
    .   with qualityLevel=B .
    .    - Function textual ID is "org.opencv.imgproc.feature.goodFeaturesToTrack"
    .   
    .   @param image Input 8-bit or floating-point 32-bit, single-channel image.
    .   @param maxCorners Maximum number of corners to return. If there are more corners than are found,
    .   the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set
    .   and all detected corners are returned.
    .   @param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
    .   parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
    .   (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
    .   quality measure less than the product are rejected. For example, if the best corner has the
    .   quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
    .   less than 15 are rejected.
    .   @param minDistance Minimum possible Euclidean distance between the returned corners.
    .   @param mask Optional region of interest. If the image is not empty (it needs to have the type
    .   CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
    .   @param blockSize Size of an average block for computing a derivative covariation matrix over each
    .   pixel neighborhood. See cornerEigenValsAndVecs .
    .   @param useHarrisDetector Parameter indicating whether to use a Harris detector (see #cornerHarris)
    .   or #cornerMinEigenVal.
    .   @param k Free parameter of the Harris detector.
    .   
    .   @return vector of detected corners.
    """
def inRange(src, threshLow, threshUp) -> retval:
    """
    .   @brief Applies a range-level threshold to each matrix element.
    .   
    .   The function applies range-level thresholding to a single- or multiple-channel matrix.
    .   It sets output pixel value to OxFF if the corresponding pixel value of input matrix is in specified range,or 0 otherwise.
    .   
    .   Input and output matrices must be CV_8UC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.inRange"
    .   
    .   @param src input matrix (CV_8UC1).
    .   @param threshLow lower boundary value.
    .   @param threshUp upper boundary value.
    .   
    .   @sa threshold
    """
def infer(name, inputs) -> retval:
    """
    .   
    
    
    
    infer(name, roi, inputs) -> retval
    .   
    
    
    
    infer(name, rois, inputs) -> retval
    .
    """
def infer2(name, in_, inputs) -> retval:
    """
    .
    """
def integral(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the integral of an image.
    .   
    .   The function calculates one or more integral images for the source image as follows:
    .   
    .   \\f[\\texttt{sum} (X,Y) =  \\sum _{x<X,y<Y}  \\texttt{image} (x,y)\\f]
    .   
    .   \\f[\\texttt{sqsum} (X,Y) =  \\sum _{x<X,y<Y}  \\texttt{image} (x,y)^2\\f]
    .   
    .   The function return integral image as \\f$(W+1)\\times (H+1)\\f$ , 32-bit integer or floating-point (32f or 64f) and
    .    integral image for squared pixel values; it is \\f$(W+1)\\times (H+)\\f$, double-precision floating-point (64f) array.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.integral"
    .   
    .   @param src input image.
    .   @param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
    .   CV_64F.
    .   @param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
    """
def kernel(op_cls):
    ...
def kernels(*args, **kwargs) -> GKernelPackage:
    ...
def kmeans(data, K, bestLabels, criteria, attempts, flags) -> retval:
    """
    .   @brief Finds centers of clusters and groups input samples around the clusters.
    .   
    .   The function kmeans implements a k-means algorithm that finds the centers of K clusters
    .   and groups the input samples around the clusters. As an output, \\f$\\texttt{bestLabels}_i\\f$
    .   contains a 0-based cluster index for the \\f$i^{th}\\f$ sample.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.kmeansND"
    .    - In case of an N-dimentional points' set given, input GMat can have the following traits:
    .   2 dimensions, a single row or column if there are N channels,
    .   or N columns if there is a single channel. Mat should have @ref CV_32F depth.
    .    - Although, if GMat with height != 1, width != 1, channels != 1 given as data, n-dimensional
    .   samples are considered given in amount of A, where A = height, n = width * channels.
    .    - In case of GMat given as data:
    .        - the output labels are returned as 1-channel GMat with sizes
    .   width = 1, height = A, where A is samples amount, or width = bestLabels.width,
    .   height = bestLabels.height if bestLabels given;
    .        - the cluster centers are returned as 1-channel GMat with sizes
    .   width = n, height = K, where n is samples' dimentionality and K is clusters' amount.
    .    - As one of possible usages, if you want to control the initial labels for each attempt
    .   by yourself, you can utilize just the core of the function. To do that, set the number
    .   of attempts to 1, initialize labels each time using a custom algorithm, pass them with the
    .   ( flags = #KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best (most-compact) clustering.
    .   
    .   @param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
    .   Function can take GArray<Point2f>, GArray<Point3f> for 2D and 3D cases or GMat for any
    .   dimentionality and channels.
    .   @param K Number of clusters to split the set by.
    .   @param bestLabels Optional input integer array that can store the supposed initial cluster indices
    .   for every sample. Used when ( flags = #KMEANS_USE_INITIAL_LABELS ) flag is set.
    .   @param criteria The algorithm termination criteria, that is, the maximum number of iterations
    .   and/or the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of
    .   the cluster centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
    .   @param attempts Flag to specify the number of times the algorithm is executed using different
    .   initial labellings. The algorithm returns the labels that yield the best compactness (see the first
    .   function return value).
    .   @param flags Flag that can take values of cv::KmeansFlags .
    .   
    .   @return
    .    - Compactness measure that is computed as
    .   \\f[\\sum _i  \\| \\texttt{samples} _i -  \\texttt{centers} _{ \\texttt{labels} _i} \\| ^2\\f]
    .   after every attempt. The best (minimum) value is chosen and the corresponding labels and the
    .   compactness value are returned by the function.
    .    - Integer array that stores the cluster indices for every sample.
    .    - Array of the cluster centers.
    
    
    
    kmeans(data, K, criteria, attempts, flags) -> retval
    .   @overload
    .   @note
    .    - Function textual ID is "org.opencv.core.kmeansNDNoInit"
    .    - #KMEANS_USE_INITIAL_LABELS flag must not be set while using this overload.
    """
def mask(src, mask) -> retval:
    """
    .   @brief Applies a mask to a matrix.
    .   
    .   The function mask set value from given matrix if the corresponding pixel value in mask matrix set to true,
    .   and set the matrix value to 0 otherwise.
    .   
    .   Supported src matrix data types are @ref CV_8UC1, @ref CV_16SC1, @ref CV_16UC1. Supported mask data type is @ref CV_8UC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.mask"
    .   @param src input matrix.
    .   @param mask input mask matrix.
    """
def max(src1, src2) -> retval:
    """
    .   @brief Calculates per-element maximum of two matrices.
    .   
    .   The function max calculates the per-element maximum of two matrices of the same size, number of channels and depth:
    .   \\f[\\texttt{dst} (I)= \\max ( \\texttt{src1} (I), \\texttt{src2} (I))\\f]
    .       where I is a multi-dimensional index of matrix elements. In case of
    .       multi-channel matrices, each channel is processed independently.
    .   Output matrix must be of the same size and depth as src1.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.max"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix of the same size and depth as src1.
    .   @sa min, compare, cmpEQ, cmpGT, cmpGE
    """
def mean(src) -> retval:
    """
    .   @brief Calculates an average (mean) of matrix elements.
    .   
    .   The function mean calculates the mean value M of matrix elements,
    .   independently for each channel, and return it.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.mean"
    .   @param src input matrix.
    .   @sa  countNonZero, min, max
    """
def medianBlur(src, ksize) -> retval:
    """
    .   @brief Blurs an image using the median filter.
    .   
    .   The function smoothes an image using the median filter with the \\f$\\texttt{ksize} \\times
    .   \\texttt{ksize}\\f$ aperture. Each channel of a multi-channel image is processed independently.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
    .   The median filter uses cv::BORDER_REPLICATE internally to cope with border pixels, see cv::BorderTypes
    .    - Function textual ID is "org.opencv.imgproc.filters.medianBlur"
    .   
    .   @param src input matrix (image)
    .   @param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
    .   @sa  boxFilter, gaussianBlur
    """
def merge3(src1, src2, src3) -> retval:
    """
    .   @brief Creates one 3-channel matrix out of 3 single-channel ones.
    .   
    .   The function merges several matrices to make a single multi-channel matrix. That is, each
    .   element of the output matrix will be a concatenation of the elements of the input matrices, where
    .   elements of i-th input matrix are treated as mv[i].channels()-element vectors.
    .   Output matrix must be of @ref CV_8UC3 type.
    .   
    .   The function split3 does the reverse operation.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transform.merge3"
    .   
    .   @param src1 first input @ref CV_8UC1 matrix to be merged.
    .   @param src2 second input @ref CV_8UC1 matrix to be merged.
    .   @param src3 third input @ref CV_8UC1 matrix to be merged.
    .   @sa merge4, split4, split3
    """
def merge4(src1, src2, src3, src4) -> retval:
    """
    .   @brief Creates one 4-channel matrix out of 4 single-channel ones.
    .   
    .   The function merges several matrices to make a single multi-channel matrix. That is, each
    .   element of the output matrix will be a concatenation of the elements of the input matrices, where
    .   elements of i-th input matrix are treated as mv[i].channels()-element vectors.
    .   Output matrix must be of @ref CV_8UC4 type.
    .   
    .   The function split4 does the reverse operation.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transform.merge4"
    .   
    .   @param src1 first input @ref CV_8UC1 matrix to be merged.
    .   @param src2 second input @ref CV_8UC1 matrix to be merged.
    .   @param src3 third input @ref CV_8UC1 matrix to be merged.
    .   @param src4 fourth input @ref CV_8UC1 matrix to be merged.
    .   @sa merge3, split4, split3
    """
def min(src1, src2) -> retval:
    """
    .   @brief Calculates per-element minimum of two matrices.
    .   
    .   The function min calculates the per-element minimum of two matrices of the same size, number of channels and depth:
    .   \\f[\\texttt{dst} (I)= \\min ( \\texttt{src1} (I), \\texttt{src2} (I))\\f]
    .       where I is a multi-dimensional index of matrix elements. In case of
    .       multi-channel matrices, each channel is processed independently.
    .   Output matrix must be of the same size and depth as src1.
    .   
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.min"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix of the same size and depth as src1.
    .   @sa max, cmpEQ, cmpLT, cmpLE
    """
def morphologyEx(*args, **kwargs) -> retval:
    """
    .   @brief Performs advanced morphological transformations.
    .   
    .   The function can perform advanced morphological transformations using an erosion and dilation as
    .   basic operations.
    .   
    .   Any of the operations can be done in-place. In case of multi-channel images, each channel is
    .   processed independently.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.imgproc.filters.morphologyEx"
    .    - The number of iterations is the number of times erosion or dilatation operation will be
    .   applied. For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to
    .   apply successively: erode -> erode -> dilate -> dilate
    .   (and not erode -> dilate -> erode -> dilate).
    .   
    .   @param src Input image.
    .   @param op Type of a morphological operation, see #MorphTypes
    .   @param kernel Structuring element. It can be created using #getStructuringElement.
    .   @param anchor Anchor position within the element. Both negative values mean that the anchor is at
    .   the kernel center.
    .   @param iterations Number of times erosion and dilation are applied.
    .   @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
    .   @param borderValue Border value in case of a constant border. The default value has a special
    .   meaning.
    .   @sa  dilate, erode, getStructuringElement
    """
def mul(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element scaled product of two matrices.
    .   
    .   The function mul calculates the per-element product of two matrices:
    .   
    .   \\f[\\texttt{dst} (I)= \\texttt{saturate} ( \\texttt{scale} \\cdot \\texttt{src1} (I)  \\cdot \\texttt{src2} (I))\\f]
    .   
    .   If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
    .   the same depth as the input matrices. The matrices can be single or multi channel.
    .   Output matrix must have the same size as input matrices.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.mul"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix of the same size and the same depth as src1.
    .   @param scale optional scale factor.
    .   @param ddepth optional depth of the output matrix.
    .   @sa add, sub, div, addWeighted
    """
def mulC(*args, **kwargs) -> retval:
    """
    .   @brief Multiplies matrix by scalar.
    .   
    .   The function mulC multiplies each element of matrix src by given scalar value:
    .   
    .   \\f[\\texttt{dst} (I)= \\texttt{saturate} (  \\texttt{src1} (I)  \\cdot \\texttt{multiplier} )\\f]
    .   
    .   The matrices can be single or multi channel. Output matrix must have the same size as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.mulC"
    .   @param src input matrix.
    .   @param multiplier factor to be multiplied.
    .   @param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
    .   @sa add, sub, div, addWeighted
    
    
    
    mulC(multiplier, src[, ddepth]) -> retval
    .
    """
def networks(*args):
    ...
def normInf(src) -> retval:
    """
    .   @brief Calculates the absolute infinite norm of a matrix.
    .   
    .   This version of normInf calculates the absolute infinite norm of src.
    .   
    .   As example for one array consider the function \\f$r(x)= \\begin{pmatrix} x \\\\ 1-x \\end{pmatrix}, x \\in [-1;1]\\f$.
    .   The \\f$ L_{\\infty} \\f$ norm for the sample value \\f$r(-1) = \\begin{pmatrix} -1 \\\\ 2 \\end{pmatrix}\\f$
    .   is calculated as follows
    .   \\f{align*}
    .       \\| r(-1) \\|_{L_\\infty} &= \\max(|-1|,|2|) = 2
    .   \\f}
    .   and for \\f$r(0.5) = \\begin{pmatrix} 0.5 \\\\ 0.5 \\end{pmatrix}\\f$ the calculation is
    .   \\f{align*}
    .       \\| r(0.5) \\|_{L_\\infty} &= \\max(|0.5|,|0.5|) = 0.5.
    .   \\f}
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.norminf"
    .   @param src input matrix.
    .   @sa normL1, normL2
    """
def normL1(src) -> retval:
    """
    .   @brief Calculates the  absolute L1 norm of a matrix.
    .   
    .   This version of normL1 calculates the absolute L1 norm of src.
    .   
    .   As example for one array consider the function \\f$r(x)= \\begin{pmatrix} x \\\\ 1-x \\end{pmatrix}, x \\in [-1;1]\\f$.
    .   The \\f$ L_{1} \\f$ norm for the sample value \\f$r(-1) = \\begin{pmatrix} -1 \\\\ 2 \\end{pmatrix}\\f$
    .   is calculated as follows
    .   \\f{align*}
    .       \\| r(-1) \\|_{L_1} &= |-1| + |2| = 3 \\\\
    .   \\f}
    .   and for \\f$r(0.5) = \\begin{pmatrix} 0.5 \\\\ 0.5 \\end{pmatrix}\\f$ the calculation is
    .   \\f{align*}
    .       \\| r(0.5) \\|_{L_1} &= |0.5| + |0.5| = 1 \\\\
    .   \\f}
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.norml1"
    .   @param src input matrix.
    .   @sa normL2, normInf
    """
def normL2(src) -> retval:
    """
    .   @brief Calculates the absolute L2 norm of a matrix.
    .   
    .   This version of normL2 calculates the absolute L2 norm of src.
    .   
    .   As example for one array consider the function \\f$r(x)= \\begin{pmatrix} x \\\\ 1-x \\end{pmatrix}, x \\in [-1;1]\\f$.
    .   The \\f$ L_{2} \\f$  norm for the sample value \\f$r(-1) = \\begin{pmatrix} -1 \\\\ 2 \\end{pmatrix}\\f$
    .   is calculated as follows
    .   \\f{align*}
    .       \\| r(-1) \\|_{L_2} &= \\sqrt{(-1)^{2} + (2)^{2}} = \\sqrt{5} \\\\
    .   \\f}
    .   and for \\f$r(0.5) = \\begin{pmatrix} 0.5 \\\\ 0.5 \\end{pmatrix}\\f$ the calculation is
    .   \\f{align*}
    .       \\| r(0.5) \\|_{L_2} &= \\sqrt{(0.5)^{2} + (0.5)^{2}} = \\sqrt{0.5} \\\\
    .   \\f}
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   @note Function textual ID is "org.opencv.core.matrixop.norml2"
    .   @param src input matrix.
    .   @sa normL1, normInf
    """
def normalize(*args, **kwargs) -> retval:
    """
    .   @brief Normalizes the norm or value range of an array.
    .   
    .   The function normalizes scale and shift the input array elements so that
    .   \\f[\\| \\texttt{dst} \\| _{L_p}= \\texttt{alpha}\\f]
    .   (where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
    .   \\f[\\min _I  \\texttt{dst} (I)= \\texttt{alpha} , \\, \\, \\max _I  \\texttt{dst} (I)= \\texttt{beta}\\f]
    .   when normType=NORM_MINMAX (for dense arrays only).
    .   
    .   @note Function textual ID is "org.opencv.core.normalize"
    .   
    .   @param src input array.
    .   @param alpha norm value to normalize to or the lower range boundary in case of the range
    .   normalization.
    .   @param beta upper range boundary in case of the range normalization; it is not used for the norm
    .   normalization.
    .   @param norm_type normalization type (see cv::NormTypes).
    .   @param ddepth when negative, the output array has the same type as src; otherwise, it has the same
    .   number of channels as src and the depth =ddepth.
    .   @sa norm, Mat::convertTo
    """
def op(op_id, in_types, out_types):
    ...
def parseSSD(*args, **kwargs) -> retval:
    """
    .   @brief Parses output of SSD network.
    .   
    .   Extracts detection information (box, confidence, label) from SSD output and
    .   filters it by given confidence and label.
    .   
    .   @note Function textual ID is "org.opencv.nn.parsers.parseSSD_BL"
    .   
    .   @param in Input CV_32F tensor with {1,1,N,7} dimensions.
    .   @param inSz Size to project detected boxes to (size of the input image).
    .   @param confidenceThreshold If confidence of the
    .   detection is smaller than confidence threshold, detection is rejected.
    .   @param filterLabel If provided (!= -1), only detections with
    .   given label will get to the output.
    .   @return a tuple with a vector of detected boxes and a vector of appropriate labels.
    
    
    
    parseSSD(in_, inSz, confidenceThreshold, alignmentToSquare, filterOutOfBounds) -> retval
    .   @brief Parses output of SSD network.
    .   
    .   Extracts detection information (box, confidence) from SSD output and
    .   filters it by given confidence and by going out of bounds.
    .   
    .   @note Function textual ID is "org.opencv.nn.parsers.parseSSD"
    .   
    .   @param in Input CV_32F tensor with {1,1,N,7} dimensions.
    .   @param inSz Size to project detected boxes to (size of the input image).
    .   @param confidenceThreshold If confidence of the
    .   detection is smaller than confidence threshold, detection is rejected.
    .   @param alignmentToSquare If provided true, bounding boxes are extended to squares.
    .   The center of the rectangle remains unchanged, the side of the square is
    .   the larger side of the rectangle.
    .   @param filterOutOfBounds If provided true, out-of-frame boxes are filtered.
    .   @return a vector of detected bounding boxes.
    """
def parseYolo(*args, **kwargs) -> retval:
    """
    .   @brief Parses output of Yolo network.
    .   
    .   Extracts detection information (box, confidence, label) from Yolo output,
    .   filters it by given confidence and performs non-maximum suppression for overlapping boxes.
    .   
    .   @note Function textual ID is "org.opencv.nn.parsers.parseYolo"
    .   
    .   @param in Input CV_32F tensor with {1,13,13,N} dimensions, N should satisfy:
    .   \\f[\\texttt{N} = (\\texttt{num_classes} + \\texttt{5}) * \\texttt{5},\\f]
    .   where num_classes - a number of classes Yolo network was trained with.
    .   @param inSz Size to project detected boxes to (size of the input image).
    .   @param confidenceThreshold If confidence of the
    .   detection is smaller than confidence threshold, detection is rejected.
    .   @param nmsThreshold Non-maximum suppression threshold which controls minimum
    .   relative box intersection area required for rejecting the box with a smaller confidence.
    .   If 1.f, nms is not performed and no boxes are rejected.
    .   @param anchors Anchors Yolo network was trained with.
    .   @note The default anchor values are specified for YOLO v2 Tiny as described in Intel Open Model Zoo
    .   <a href="https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v2-tiny-tf/yolo-v2-tiny-tf.md">documentation</a>.
    .   @return a tuple with a vector of detected boxes and a vector of appropriate labels.
    """
def phase(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the rotation angle of 2D vectors.
    .   
    .   The function cv::phase calculates the rotation angle of each 2D vector that
    .   is formed from the corresponding elements of x and y :
    .   \\f[\\texttt{angle} (I) =  \\texttt{atan2} ( \\texttt{y} (I), \\texttt{x} (I))\\f]
    .   
    .   The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
    .   the corresponding angle(I) is set to 0.
    .   @param x input floating-point array of x-coordinates of 2D vectors.
    .   @param y input array of y-coordinates of 2D vectors; it must have the
    .   same size and the same type as x.
    .   @param angleInDegrees when true, the function calculates the angle in
    .   degrees, otherwise, they are measured in radians.
    .   @return array of vector angles; it has the same size and same type as x.
    """
def polarToCart(*args, **kwargs) -> retval:
    """
    .   @brief Calculates x and y coordinates of 2D vectors from their magnitude and angle.
    .   
    .   The function polarToCart calculates the Cartesian coordinates of each 2D
    .   vector represented by the corresponding elements of magnitude and angle:
    .   \\f[\\begin{array}{l} \\texttt{x} (I) =  \\texttt{magnitude} (I) \\cos ( \\texttt{angle} (I)) \\\\ \\texttt{y} (I) =  \\texttt{magnitude} (I) \\sin ( \\texttt{angle} (I)) \\\\ \\end{array}\\f]
    .   
    .   The relative accuracy of the estimated coordinates is about 1e-6.
    .   
    .   First output is a matrix of x-coordinates of 2D vectors.
    .   Second output is a matrix of y-coordinates of 2D vectors.
    .   Both output must have the same size and depth as input matrices.
    .   
    .   @note Function textual ID is "org.opencv.core.math.polarToCart"
    .   
    .   @param magnitude input floating-point @ref CV_32FC1 matrix (1xN) of magnitudes of 2D vectors;
    .   @param angle input floating-point @ref CV_32FC1 matrix (1xN) of angles of 2D vectors.
    .   @param angleInDegrees when true, the input angles are measured in
    .   degrees, otherwise, they are measured in radians.
    .   @sa cartToPolar, exp, log, pow, sqrt
    """
def register(mname):
    ...
def remap(*args, **kwargs) -> retval:
    """
    .   @brief Applies a generic geometrical transformation to an image.
    .   
    .   The function remap transforms the source image using the specified map:
    .   
    .   \\f[\\texttt{dst} (x,y) =  \\texttt{src} (map_x(x,y),map_y(x,y))\\f]
    .   
    .   where values of pixels with non-integer coordinates are computed using one of available
    .   interpolation methods. \\f$map_x\\f$ and \\f$map_y\\f$ can be encoded as separate floating-point maps
    .   in \\f$map_1\\f$ and \\f$map_2\\f$ respectively, or interleaved floating-point maps of \\f$(x,y)\\f$ in
    .   \\f$map_1\\f$, or fixed-point maps created by using convertMaps. The reason you might want to
    .   convert from floating to fixed-point representations of a map is that they can yield much faster
    .   (\\~2x) remapping operations. In the converted case, \\f$map_1\\f$ contains pairs (cvFloor(x),
    .   cvFloor(y)) and \\f$map_2\\f$ contains indices in a table of interpolation coefficients.
    .   Output image must be of the same size and depth as input one.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transform.remap"
    .    - Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
    .   
    .   @param src Source image.
    .   @param map1 The first map of either (x,y) points or just x values having the type CV_16SC2,
    .   CV_32FC1, or CV_32FC2.
    .   @param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
    .   if map1 is (x,y) points), respectively.
    .   @param interpolation Interpolation method (see cv::InterpolationFlags). The methods #INTER_AREA
    .   and #INTER_LINEAR_EXACT are not supported by this function.
    .   @param borderMode Pixel extrapolation method (see cv::BorderTypes). When
    .   borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image that
    .   corresponds to the "outliers" in the source image are not modified by the function.
    .   @param borderValue Value used in case of a constant border. By default, it is 0.
    """
def resize(*args, **kwargs) -> retval:
    """
    .   @brief Resizes an image.
    .   
    .   The function resizes the image src down to or up to the specified size.
    .   
    .   Output image size will have the size dsize (when dsize is non-zero) or the size computed from
    .   src.size(), fx, and fy; the depth of output is the same as of src.
    .   
    .   If you want to resize src so that it fits the pre-created dst,
    .   you may call the function as follows:
    .   @code
    .       // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    .       resize(src, dst, dst.size(), 0, 0, interpolation);
    .   @endcode
    .   If you want to decimate the image by factor of 2 in each direction, you can call the function this
    .   way:
    .   @code
    .       // specify fx and fy and let the function compute the destination image size.
    .       resize(src, dst, Size(), 0.5, 0.5, interpolation);
    .   @endcode
    .   To shrink an image, it will generally look best with cv::INTER_AREA interpolation, whereas to
    .   enlarge an image, it will generally look best with cv::INTER_CUBIC (slow) or cv::INTER_LINEAR
    .   (faster but still looks OK).
    .   
    .   @note Function textual ID is "org.opencv.imgproc.transform.resize"
    .   
    .   @param src input image.
    .   @param dsize output image size; if it equals zero, it is computed as:
    .    \\f[\\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\\f]
    .    Either dsize or both fx and fy must be non-zero.
    .   @param fx scale factor along the horizontal axis; when it equals 0, it is computed as
    .   \\f[\\texttt{(double)dsize.width/src.cols}\\f]
    .   @param fy scale factor along the vertical axis; when it equals 0, it is computed as
    .   \\f[\\texttt{(double)dsize.height/src.rows}\\f]
    .   @param interpolation interpolation method, see cv::InterpolationFlags
    .   
    .   @sa  warpAffine, warpPerspective, remap, resizeP
    """
def select(src1, src2, mask) -> retval:
    """
    .   @brief Select values from either first or second of input matrices by given mask.
    .   The function set to the output matrix either the value from the first input matrix if corresponding value of mask matrix is 255,
    .    or value from the second input matrix (if value of mask matrix set to 0).
    .   
    .   Input mask matrix must be of @ref CV_8UC1 type, two other inout matrices and output matrix should be of the same type. The size should
    .   be the same for all input and output matrices.
    .   Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.pixelwise.select"
    .   
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    .   @param mask mask input matrix.
    """
def sepFilter(*args, **kwargs) -> retval:
    """
    .   @brief Applies a separable linear filter to a matrix(image).
    .   
    .   The function applies a separable linear filter to the matrix. That is, first, every row of src is
    .   filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
    .   kernel kernelY. The final result is returned.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   Output image must have the same type, size, and number of channels as the input image.
    .   @note
    .    - In case of floating-point computation, rounding to nearest even is procedeed
    .   if hardware supports it (if not - to nearest value).
    .    - Function textual ID is "org.opencv.imgproc.filters.sepfilter"
    .   @param src Source image.
    .   @param ddepth desired depth of the destination image (the following combinations of src.depth() and ddepth are supported:
    .   
    .           src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    .           src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
    .           src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
    .           src.depth() = CV_64F, ddepth = -1/CV_64F
    .   
    .   when ddepth=-1, the output image will have the same depth as the source)
    .   @param kernelX Coefficients for filtering each row.
    .   @param kernelY Coefficients for filtering each column.
    .   @param anchor Anchor position within the kernel. The default value \\f$(-1,-1)\\f$ means that the anchor
    .   is at the kernel center.
    .   @param delta Value added to the filtered results before storing them.
    .   @param borderType Pixel extrapolation method, see cv::BorderTypes
    .   @param borderValue border value in case of constant border type
    .   @sa  boxFilter, gaussianBlur, medianBlur
    """
def split3(src) -> retval:
    """
    .   @brief Divides a 3-channel matrix into 3 single-channel matrices.
    .   
    .   The function splits a 3-channel matrix into 3 single-channel matrices:
    .   \\f[\\texttt{mv} [c](I) =  \\texttt{src} (I)_c\\f]
    .   
    .   All output matrices must be of @ref CV_8UC1 type.
    .   
    .   The function merge3 does the reverse operation.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transform.split3"
    .   
    .   @param src input @ref CV_8UC3 matrix.
    .   @sa split4, merge3, merge4
    """
def split4(src) -> retval:
    """
    .   @brief Divides a 4-channel matrix into 4 single-channel matrices.
    .   
    .   The function splits a 4-channel matrix into 4 single-channel matrices:
    .   \\f[\\texttt{mv} [c](I) =  \\texttt{src} (I)_c\\f]
    .   
    .   All output matrices must be of @ref CV_8UC1 type.
    .   
    .   The function merge4 does the reverse operation.
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transform.split4"
    .   
    .   @param src input @ref CV_8UC4 matrix.
    .   @sa split3, merge3, merge4
    """
def sqrt(src) -> retval:
    """
    .   @brief Calculates a square root of array elements.
    .   
    .   The function cv::gapi::sqrt calculates a square root of each input array element.
    .   In case of multi-channel arrays, each channel is processed
    .   independently. The accuracy is approximately the same as of the built-in
    .   std::sqrt .
    .   @param src input floating-point array.
    .   @return output array of the same size and type as src.
    """
def sub(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element difference between two matrices.
    .   
    .   The function sub calculates difference between two matrices, when both matrices have the same size and the same number of
    .   channels:
    .       \\f[\\texttt{dst}(I) =   \\texttt{src1}(I) -  \\texttt{src2}(I)\\f]
    .   
    .   The function can be replaced with matrix expressions:
    .   \\f[\\texttt{dst} =   \\texttt{src1} -  \\texttt{src2}\\f]
    .   
    .   The input matrices and the output matrix can all have the same or different depths. For example, you
    .   can subtract two 8-bit unsigned matrices store the result as a 16-bit signed matrix.
    .   Depth of the output matrix is determined by the ddepth parameter.
    .   If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
    .   the same depth as the input matrices. The matrices can be single or multi channel.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.sub"
    .   @param src1 first input matrix.
    .   @param src2 second input matrix.
    .   @param ddepth optional depth of the output matrix.
    .   @sa  add, addC
    """
def subC(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element difference between matrix and given scalar.
    .   
    .   The function can be replaced with matrix expressions:
    .       \\f[\\texttt{dst} =  \\texttt{src} - \\texttt{c}\\f]
    .   
    .   Depth of the output matrix is determined by the ddepth parameter.
    .   If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
    .   The matrices can be single or multi channel. Output matrix must have the same size as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.subC"
    .   @param src first input matrix.
    .   @param c scalar value to subtracted.
    .   @param ddepth optional depth of the output matrix.
    .   @sa  add, addC, subRC
    """
def subRC(*args, **kwargs) -> retval:
    """
    .   @brief Calculates the per-element difference between given scalar and the matrix.
    .   
    .   The function can be replaced with matrix expressions:
    .       \\f[\\texttt{dst} =  \\texttt{c} - \\texttt{src}\\f]
    .   
    .   Depth of the output matrix is determined by the ddepth parameter.
    .   If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
    .   The matrices can be single or multi channel. Output matrix must have the same size as src.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.math.subRC"
    .   @param c scalar value to subtract from.
    .   @param src input matrix to be subtracted.
    .   @param ddepth optional depth of the output matrix.
    .   @sa  add, addC, subC
    """
def sum(src) -> retval:
    """
    .   @brief Calculates sum of all matrix elements.
    .   
    .   The function sum calculates sum of all matrix elements, independently for each channel.
    .   
    .   Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.sum"
    .   @param src input matrix.
    .   @sa countNonZero, mean, min, max
    """
def threshold(src, thresh, maxval, type) -> retval:
    """
    .   @brief Applies a fixed-level threshold to each matrix element.
    .   
    .   The function applies fixed-level thresholding to a single- or multiple-channel matrix.
    .   The function is typically used to get a bi-level (binary) image out of a grayscale image ( cmp functions could be also used for
    .   this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
    .   values. There are several types of thresholding supported by the function. They are determined by
    .   type parameter.
    .   
    .   Also, the special values cv::THRESH_OTSU or cv::THRESH_TRIANGLE may be combined with one of the
    .   above values. In these cases, the function determines the optimal threshold value using the Otsu's
    .   or Triangle algorithm and uses it instead of the specified thresh . The function returns the
    .   computed threshold value in addititon to thresholded matrix.
    .   The Otsu's and Triangle methods are implemented only for 8-bit matrices.
    .   
    .   Input image should be single channel only in case of cv::THRESH_OTSU or cv::THRESH_TRIANGLE flags.
    .   Output matrix must be of the same size and depth as src.
    .   
    .   @note Function textual ID is "org.opencv.core.matrixop.threshold"
    .   
    .   @param src input matrix (@ref CV_8UC1, @ref CV_8UC3, or @ref CV_32FC1).
    .   @param thresh threshold value.
    .   @param maxval maximum value to use with the cv::THRESH_BINARY and cv::THRESH_BINARY_INV thresholding
    .   types.
    .   @param type thresholding type (see the cv::ThresholdTypes).
    .   
    .   @sa min, max, cmpGT, cmpLE, cmpGE, cmpLT
    
    
    
    threshold(src, maxval, type) -> retval
    .   @overload
    .   This function applicable for all threshold types except CV_THRESH_OTSU and CV_THRESH_TRIANGLE
    .   @note Function textual ID is "org.opencv.core.matrixop.thresholdOT"
    """
def transpose(src) -> retval:
    """
    .   @brief Transposes a matrix.
    .   
    .   The function transposes the matrix:
    .   \\f[\\texttt{dst} (i,j) =  \\texttt{src} (j,i)\\f]
    .   
    .   @note
    .    - Function textual ID is "org.opencv.core.transpose"
    .    - No complex conjugation is done in case of a complex matrix. It should be done separately if needed.
    .   
    .   @param src input array.
    """
def warpAffine(*args, **kwargs) -> retval:
    """
    .   @brief Applies an affine transformation to an image.
    .   
    .   The function warpAffine transforms the source image using the specified matrix:
    .   
    .   \\f[\\texttt{dst} (x,y) =  \\texttt{src} ( \\texttt{M} _{11} x +  \\texttt{M} _{12} y +  \\texttt{M} _{13}, \\texttt{M} _{21} x +  \\texttt{M} _{22} y +  \\texttt{M} _{23})\\f]
    .   
    .   when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
    .   with #invertAffineTransform and then put in the formula above instead of M. The function cannot
    .   operate in-place.
    .   
    .   @param src input image.
    .   @param M \\f$2\\times 3\\f$ transformation matrix.
    .   @param dsize size of the output image.
    .   @param flags combination of interpolation methods (see #InterpolationFlags) and the optional
    .   flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
    .   \\f$\\texttt{dst}\\rightarrow\\texttt{src}\\f$ ).
    .   @param borderMode pixel extrapolation method (see #BorderTypes);
    .   borderMode=#BORDER_TRANSPARENT isn't supported
    .   @param borderValue value used in case of a constant border; by default, it is 0.
    .   
    .   @sa  warpPerspective, resize, remap, getRectSubPix, transform
    """
def warpPerspective(*args, **kwargs) -> retval:
    """
    .   @brief Applies a perspective transformation to an image.
    .   
    .   The function warpPerspective transforms the source image using the specified matrix:
    .   
    .   \\f[\\texttt{dst} (x,y) =  \\texttt{src} \\left ( \\frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
    .        \\frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \\right )\\f]
    .   
    .   when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
    .   and then put in the formula above instead of M. The function cannot operate in-place.
    .   
    .   @param src input image.
    .   @param M \\f$3\\times 3\\f$ transformation matrix.
    .   @param dsize size of the output image.
    .   @param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the
    .   optional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (
    .   \\f$\\texttt{dst}\\rightarrow\\texttt{src}\\f$ ).
    .   @param borderMode pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).
    .   @param borderValue value used in case of a constant border; by default, it equals 0.
    .   
    .   @sa  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
    """
CV_ANY: int = 15
CV_BOOL: int = 0
CV_DOUBLE: int = 3
CV_DRAW_PRIM: int = 14
CV_FLOAT: int = 4
CV_GMAT: int = 13
CV_INT: int = 1
CV_INT64: int = 2
CV_MAT: int = 12
CV_POINT: int = 6
CV_POINT2F: int = 7
CV_POINT3F: int = 8
CV_RECT: int = 10
CV_SCALAR: int = 11
CV_SIZE: int = 9
CV_STRING: int = 5
STEREO_OUTPUT_FORMAT_DEPTH_16F: int = 0
STEREO_OUTPUT_FORMAT_DEPTH_32F: int = 1
STEREO_OUTPUT_FORMAT_DEPTH_FLOAT16: int = 0
STEREO_OUTPUT_FORMAT_DEPTH_FLOAT32: int = 1
STEREO_OUTPUT_FORMAT_DISPARITY_16Q_10_5: int = 2
STEREO_OUTPUT_FORMAT_DISPARITY_16Q_11_4: int = 3
STEREO_OUTPUT_FORMAT_DISPARITY_FIXED16_11_5: int = 2
STEREO_OUTPUT_FORMAT_DISPARITY_FIXED16_12_4: int = 3
StereoOutputFormat_DEPTH_16F: int = 0
StereoOutputFormat_DEPTH_32F: int = 1
StereoOutputFormat_DEPTH_FLOAT16: int = 0
StereoOutputFormat_DEPTH_FLOAT32: int = 1
StereoOutputFormat_DISPARITY_16Q_10_5: int = 2
StereoOutputFormat_DISPARITY_16Q_11_4: int = 3
StereoOutputFormat_DISPARITY_FIXED16_11_5: int = 2
StereoOutputFormat_DISPARITY_FIXED16_12_4: int = 3
