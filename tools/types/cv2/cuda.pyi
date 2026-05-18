from __future__ import annotations
__all__: list[str] = ['BufferPool', 'DEVICE_INFO_COMPUTE_MODE_DEFAULT', 'DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE', 'DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE_PROCESS', 'DEVICE_INFO_COMPUTE_MODE_PROHIBITED', 'DYNAMIC_PARALLELISM', 'DeviceInfo', 'DeviceInfo_ComputeModeDefault', 'DeviceInfo_ComputeModeExclusive', 'DeviceInfo_ComputeModeExclusiveProcess', 'DeviceInfo_ComputeModeProhibited', 'EVENT_BLOCKING_SYNC', 'EVENT_DEFAULT', 'EVENT_DISABLE_TIMING', 'EVENT_INTERPROCESS', 'Event', 'Event_BLOCKING_SYNC', 'Event_DEFAULT', 'Event_DISABLE_TIMING', 'Event_INTERPROCESS', 'Event_elapsedTime', 'FEATURE_SET_COMPUTE_10', 'FEATURE_SET_COMPUTE_11', 'FEATURE_SET_COMPUTE_12', 'FEATURE_SET_COMPUTE_13', 'FEATURE_SET_COMPUTE_20', 'FEATURE_SET_COMPUTE_21', 'FEATURE_SET_COMPUTE_30', 'FEATURE_SET_COMPUTE_32', 'FEATURE_SET_COMPUTE_35', 'FEATURE_SET_COMPUTE_50', 'GLOBAL_ATOMICS', 'GpuData', 'GpuMat', 'GpuMatND', 'GpuMat_defaultAllocator', 'GpuMat_setDefaultAllocator', 'HOST_MEM_PAGE_LOCKED', 'HOST_MEM_SHARED', 'HOST_MEM_WRITE_COMBINED', 'HostMem', 'HostMem_PAGE_LOCKED', 'HostMem_SHARED', 'HostMem_WRITE_COMBINED', 'NATIVE_DOUBLE', 'SHARED_ATOMICS', 'Stream', 'Stream_Null', 'TargetArchs', 'TargetArchs_has', 'TargetArchs_hasBin', 'TargetArchs_hasEqualOrGreater', 'TargetArchs_hasEqualOrGreaterBin', 'TargetArchs_hasEqualOrGreaterPtx', 'TargetArchs_hasEqualOrLessPtx', 'TargetArchs_hasPtx', 'WARP_SHUFFLE_FUNCTIONS', 'createContinuous', 'ensureSizeIsEnough', 'fastNlMeansDenoising', 'fastNlMeansDenoisingColored', 'getCudaEnabledDeviceCount', 'getDevice', 'nonLocalMeans', 'printCudaDeviceInfo', 'printShortCudaDeviceInfo', 'registerPageLocked', 'resetDevice', 'setBufferPoolConfig', 'setBufferPoolUsage', 'setDevice', 'unregisterPageLocked']
class BufferPool:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def getAllocator() -> retval:
        """
        .
        """
    @staticmethod
    def getBuffer(rows, cols, type) -> retval:
        """
        .   
        
        
        
        getBuffer(size, type) -> retval
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class DeviceInfo:
    @staticmethod
    def ECCEnabled() -> retval:
        """
        .
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def asyncEngineCount() -> retval:
        """
        .
        """
    @staticmethod
    def canMapHostMemory() -> retval:
        """
        .
        """
    @staticmethod
    def clockRate() -> retval:
        """
        .
        """
    @staticmethod
    def computeMode() -> retval:
        """
        .
        """
    @staticmethod
    def concurrentKernels() -> retval:
        """
        .
        """
    @staticmethod
    def deviceID() -> retval:
        """
        .   @brief Returns system index of the CUDA device starting with 0.
        """
    @staticmethod
    def freeMemory() -> retval:
        """
        .
        """
    @staticmethod
    def integrated() -> retval:
        """
        .
        """
    @staticmethod
    def isCompatible() -> retval:
        """
        .   @brief Checks the CUDA module and device compatibility.
        .   
        .       This function returns true if the CUDA module can be run on the specified device. Otherwise, it
        .       returns false .
        """
    @staticmethod
    def kernelExecTimeoutEnabled() -> retval:
        """
        .
        """
    @staticmethod
    def l2CacheSize() -> retval:
        """
        .
        """
    @staticmethod
    def majorVersion() -> retval:
        """
        .
        """
    @staticmethod
    def maxGridSize() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurface1D() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurface1DLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurface2D() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurface2DLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurface3D() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurfaceCubemap() -> retval:
        """
        .
        """
    @staticmethod
    def maxSurfaceCubemapLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture1D() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture1DLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture1DLinear() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture1DMipmap() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture2D() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture2DGather() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture2DLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture2DLinear() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture2DMipmap() -> retval:
        """
        .
        """
    @staticmethod
    def maxTexture3D() -> retval:
        """
        .
        """
    @staticmethod
    def maxTextureCubemap() -> retval:
        """
        .
        """
    @staticmethod
    def maxTextureCubemapLayered() -> retval:
        """
        .
        """
    @staticmethod
    def maxThreadsDim() -> retval:
        """
        .
        """
    @staticmethod
    def maxThreadsPerBlock() -> retval:
        """
        .
        """
    @staticmethod
    def maxThreadsPerMultiProcessor() -> retval:
        """
        .
        """
    @staticmethod
    def memPitch() -> retval:
        """
        .
        """
    @staticmethod
    def memoryBusWidth() -> retval:
        """
        .
        """
    @staticmethod
    def memoryClockRate() -> retval:
        """
        .
        """
    @staticmethod
    def minorVersion() -> retval:
        """
        .
        """
    @staticmethod
    def multiProcessorCount() -> retval:
        """
        .
        """
    @staticmethod
    def pciBusID() -> retval:
        """
        .
        """
    @staticmethod
    def pciDeviceID() -> retval:
        """
        .
        """
    @staticmethod
    def pciDomainID() -> retval:
        """
        .
        """
    @staticmethod
    def queryMemory(totalMemory, freeMemory) -> None:
        """
        .
        """
    @staticmethod
    def regsPerBlock() -> retval:
        """
        .
        """
    @staticmethod
    def sharedMemPerBlock() -> retval:
        """
        .
        """
    @staticmethod
    def surfaceAlignment() -> retval:
        """
        .
        """
    @staticmethod
    def tccDriver() -> retval:
        """
        .
        """
    @staticmethod
    def textureAlignment() -> retval:
        """
        .
        """
    @staticmethod
    def texturePitchAlignment() -> retval:
        """
        .
        """
    @staticmethod
    def totalConstMem() -> retval:
        """
        .
        """
    @staticmethod
    def totalGlobalMem() -> retval:
        """
        .
        """
    @staticmethod
    def totalMemory() -> retval:
        """
        .
        """
    @staticmethod
    def unifiedAddressing() -> retval:
        """
        .
        """
    @staticmethod
    def warpSize() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Event:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def elapsedTime(start, end) -> retval:
        """
        .
        """
    @staticmethod
    def queryIfComplete() -> retval:
        """
        .
        """
    @staticmethod
    def record(*args, **kwargs) -> None:
        """
        .
        """
    @staticmethod
    def waitForCompletion() -> None:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GpuData:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GpuMat:
    class Allocator:
        @staticmethod
        def __new__(type, *args, **kwargs):
            """
            Create and return a new object.  See help(type) for accurate signature.
            """
        def __repr__(self):
            """
            Return repr(self).
            """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def adjustROI(dtop, dbottom, dleft, dright) -> retval:
        """
        .
        """
    @staticmethod
    def assignTo(*args, **kwargs) -> None:
        """
        .
        """
    @staticmethod
    def channels() -> retval:
        """
        .
        """
    @staticmethod
    def clone() -> retval:
        """
        .
        """
    @staticmethod
    def col(x) -> retval:
        """
        .
        """
    @staticmethod
    def colRange(startcol, endcol) -> retval:
        """
        .   
        
        
        
        colRange(r) -> retval
        .
        """
    @staticmethod
    def convertTo(*args, **kwargs) -> dst:
        """
        .   
        
        
        
        convertTo(rtype, stream[, dst]) -> dst
        .   
        
        
        
        convertTo(rtype, alpha[, dst[, beta]]) -> dst
        .   
        
        
        
        convertTo(rtype, alpha, stream[, dst]) -> dst
        .   
        
        
        
        convertTo(rtype, alpha, beta, stream[, dst]) -> dst
        .
        """
    @staticmethod
    def copyTo(*args, **kwargs) -> dst:
        """
        .   
        
        
        
        copyTo(stream[, dst]) -> dst
        .   
        
        
        
        copyTo(mask[, dst]) -> dst
        .   
        
        
        
        copyTo(mask, stream[, dst]) -> dst
        .
        """
    @staticmethod
    def create(rows, cols, type) -> None:
        """
        .   
        
        
        
        create(size, type) -> None
        .
        """
    @staticmethod
    def cudaPtr() -> retval:
        """
        .
        """
    @staticmethod
    def defaultAllocator() -> retval:
        """
        .
        """
    @staticmethod
    def depth() -> retval:
        """
        .
        """
    @staticmethod
    def download(*args, **kwargs) -> dst:
        """
        .   @brief Performs data download from GpuMat (Blocking call)
        .   
        .       This function copies data from device memory to host memory. As being a blocking call, it is
        .       guaranteed that the copy operation is finished when this function returns.
        
        
        
        download(stream[, dst]) -> dst
        .   @brief Performs data download from GpuMat (Non-Blocking call)
        .   
        .       This function copies data from device memory to host memory. As being a non-blocking call, this
        .       function may return even if the copy operation is not finished.
        .   
        .       The copy operation may be overlapped with operations in other non-default streams if \\p stream is
        .       not the default stream and \\p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
        """
    @staticmethod
    def elemSize() -> retval:
        """
        .
        """
    @staticmethod
    def elemSize1() -> retval:
        """
        .
        """
    @staticmethod
    def empty() -> retval:
        """
        .
        """
    @staticmethod
    def isContinuous() -> retval:
        """
        .
        """
    @staticmethod
    def locateROI(wholeSize, ofs) -> None:
        """
        .
        """
    @staticmethod
    def release() -> None:
        """
        .
        """
    @staticmethod
    def reshape(*args, **kwargs) -> retval:
        """
        .
        """
    @staticmethod
    def row(y) -> retval:
        """
        .
        """
    @staticmethod
    def rowRange(startrow, endrow) -> retval:
        """
        .   
        
        
        
        rowRange(r) -> retval
        .
        """
    @staticmethod
    def setDefaultAllocator(allocator) -> None:
        """
        .
        """
    @staticmethod
    def setTo(s) -> retval:
        """
        .   
        
        
        
        setTo(s, stream) -> retval
        .   
        
        
        
        setTo(s, mask) -> retval
        .   
        
        
        
        setTo(s, mask, stream) -> retval
        .
        """
    @staticmethod
    def size() -> retval:
        """
        .
        """
    @staticmethod
    def step1() -> retval:
        """
        .
        """
    @staticmethod
    def swap(mat) -> None:
        """
        .
        """
    @staticmethod
    def type() -> retval:
        """
        .
        """
    @staticmethod
    def updateContinuityFlag() -> None:
        """
        .
        """
    @staticmethod
    def upload(arr) -> None:
        """
        .   @brief Performs data upload to GpuMat (Blocking call)
        .   
        .       This function copies data from host memory to device memory. As being a blocking call, it is
        .       guaranteed that the copy operation is finished when this function returns.
        
        
        
        upload(arr, stream) -> None
        .   @brief Performs data upload to GpuMat (Non-Blocking call)
        .   
        .       This function copies data from host memory to device memory. As being a non-blocking call, this
        .       function may return even if the copy operation is not finished.
        .   
        .       The copy operation may be overlapped with operations in other non-default streams if \\p stream is
        .       not the default stream and \\p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GpuMatND:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class HostMem:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def channels() -> retval:
        """
        .
        """
    @staticmethod
    def clone() -> retval:
        """
        .
        """
    @staticmethod
    def create(rows, cols, type) -> None:
        """
        .
        """
    @staticmethod
    def createMatHeader() -> retval:
        """
        .
        """
    @staticmethod
    def depth() -> retval:
        """
        .
        """
    @staticmethod
    def elemSize() -> retval:
        """
        .
        """
    @staticmethod
    def elemSize1() -> retval:
        """
        .
        """
    @staticmethod
    def empty() -> retval:
        """
        .
        """
    @staticmethod
    def isContinuous() -> retval:
        """
        .   @brief Maps CPU memory to GPU address space and creates the cuda::GpuMat header without reference counting
        .       for it.
        .   
        .       This can be done only if memory was allocated with the SHARED flag and if it is supported by the
        .       hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which
        .       eliminates an extra copy.
        """
    @staticmethod
    def reshape(*args, **kwargs) -> retval:
        """
        .
        """
    @staticmethod
    def size() -> retval:
        """
        .
        """
    @staticmethod
    def step1() -> retval:
        """
        .
        """
    @staticmethod
    def swap(b) -> None:
        """
        .
        """
    @staticmethod
    def type() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Stream:
    @staticmethod
    def Null() -> retval:
        """
        .   @brief Adds a callback to be called on the host after all currently enqueued items in the stream have
        .       completed.
        .   
        .       @note Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization
        .       that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
        .       Callbacks without a mandated order (in independent streams) execute in undefined order and may be
        .       serialized.
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def cudaPtr() -> retval:
        """
        .
        """
    @staticmethod
    def queryIfComplete() -> retval:
        """
        .   @brief Returns true if the current stream queue is finished. Otherwise, it returns false.
        """
    @staticmethod
    def waitEvent(event) -> None:
        """
        .   @brief Makes a compute stream wait on an event.
        """
    @staticmethod
    def waitForCompletion() -> None:
        """
        .   @brief Blocks the current CPU thread until all operations in the stream are complete.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class TargetArchs:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def has(major, minor) -> retval:
        """
        .   @brief There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA
        .       code for the given architecture(s):
        .   
        .       @param major Major compute capability version.
        .       @param minor Minor compute capability version.
        """
    @staticmethod
    def hasBin(major, minor) -> retval:
        """
        .
        """
    @staticmethod
    def hasEqualOrGreater(major, minor) -> retval:
        """
        .
        """
    @staticmethod
    def hasEqualOrGreaterBin(major, minor) -> retval:
        """
        .
        """
    @staticmethod
    def hasEqualOrGreaterPtx(major, minor) -> retval:
        """
        .
        """
    @staticmethod
    def hasEqualOrLessPtx(major, minor) -> retval:
        """
        .
        """
    @staticmethod
    def hasPtx(major, minor) -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def Event_elapsedTime(start, end) -> retval:
    """
    .
    """
def GpuMat_defaultAllocator() -> retval:
    """
    .
    """
def GpuMat_setDefaultAllocator(allocator) -> None:
    """
    .
    """
def Stream_Null() -> retval:
    """
    .   @brief Adds a callback to be called on the host after all currently enqueued items in the stream have
    .       completed.
    .   
    .       @note Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization
    .       that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
    .       Callbacks without a mandated order (in independent streams) execute in undefined order and may be
    .       serialized.
    """
def TargetArchs_has(major, minor) -> retval:
    """
    .   @brief There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA
    .       code for the given architecture(s):
    .   
    .       @param major Major compute capability version.
    .       @param minor Minor compute capability version.
    """
def TargetArchs_hasBin(major, minor) -> retval:
    """
    .
    """
def TargetArchs_hasEqualOrGreater(major, minor) -> retval:
    """
    .
    """
def TargetArchs_hasEqualOrGreaterBin(major, minor) -> retval:
    """
    .
    """
def TargetArchs_hasEqualOrGreaterPtx(major, minor) -> retval:
    """
    .
    """
def TargetArchs_hasEqualOrLessPtx(major, minor) -> retval:
    """
    .
    """
def TargetArchs_hasPtx(major, minor) -> retval:
    """
    .
    """
def createContinuous(*args, **kwargs) -> arr:
    """
    .   @brief Creates a continuous matrix.
    .   
    .   @param rows Row count.
    .   @param cols Column count.
    .   @param type Type of the matrix.
    .   @param arr Destination matrix. This parameter changes only if it has a proper type and area (
    .   \\f$\\texttt{rows} \\times \\texttt{cols}\\f$ ).
    .   
    .   Matrix is called continuous if its elements are stored continuously, that is, without gaps at the
    .   end of each row.
    """
def ensureSizeIsEnough(*args, **kwargs) -> arr:
    """
    .   @brief Ensures that the size of a matrix is big enough and the matrix has a proper type.
    .   
    .   @param rows Minimum desired number of rows.
    .   @param cols Minimum desired number of columns.
    .   @param type Desired matrix type.
    .   @param arr Destination matrix.
    .   
    .   The function does not reallocate memory if the matrix has proper attributes already.
    """
def fastNlMeansDenoising(*args, **kwargs) -> dst:
    """
    .   @brief Perform image denoising using Non-local Means Denoising algorithm
    .   <http://www.ipol.im/pub/algo/bcm_non_local_means_denoising> with several computational
    .   optimizations. Noise expected to be a gaussian white noise
    .   
    .   @param src Input 8-bit 1-channel, 2-channel or 3-channel image.
    .   @param dst Output image with the same size and type as src .
    .   @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
    .   removes image details, smaller h value preserves details but also preserves some noise
    .   @param search_window Size in pixels of the window that is used to compute weighted average for
    .   given pixel. Should be odd. Affect performance linearly: greater search_window - greater
    .   denoising time. Recommended value 21 pixels
    .   @param block_size Size in pixels of the template patch that is used to compute weights. Should be
    .   odd. Recommended value 7 pixels
    .   @param stream Stream for the asynchronous invocations.
    .   
    .   This function expected to be applied to grayscale images. For colored images look at
    .   FastNonLocalMeansDenoising::labMethod.
    .   
    .   @sa
    .      fastNlMeansDenoising
    """
def fastNlMeansDenoisingColored(*args, **kwargs) -> dst:
    """
    .   @brief Modification of fastNlMeansDenoising function for colored images
    .   
    .   @param src Input 8-bit 3-channel image.
    .   @param dst Output image with the same size and type as src .
    .   @param h_luminance Parameter regulating filter strength. Big h value perfectly removes noise but
    .   also removes image details, smaller h value preserves details but also preserves some noise
    .   @param photo_render float The same as h but for color components. For most images value equals 10 will be
    .   enough to remove colored noise and do not distort colors
    .   @param search_window Size in pixels of the window that is used to compute weighted average for
    .   given pixel. Should be odd. Affect performance linearly: greater search_window - greater
    .   denoising time. Recommended value 21 pixels
    .   @param block_size Size in pixels of the template patch that is used to compute weights. Should be
    .   odd. Recommended value 7 pixels
    .   @param stream Stream for the asynchronous invocations.
    .   
    .   The function converts image to CIELAB colorspace and then separately denoise L and AB components
    .   with given h parameters using FastNonLocalMeansDenoising::simpleMethod function.
    .   
    .   @sa
    .      fastNlMeansDenoisingColored
    """
def getCudaEnabledDeviceCount() -> retval:
    """
    .   @brief Returns the number of installed CUDA-enabled devices.
    .   
    .   Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support,
    .   this function returns 0. If the CUDA driver is not installed, or is incompatible, this function
    .   returns -1.
    """
def getDevice() -> retval:
    """
    .   @brief Returns the current device index set by cuda::setDevice or initialized by default.
    """
def nonLocalMeans(*args, **kwargs) -> dst:
    """
    .   @brief Performs pure non local means denoising without any simplification, and thus it is not fast.
    .   
    .   @param src Source image. Supports only CV_8UC1, CV_8UC2 and CV_8UC3.
    .   @param dst Destination image.
    .   @param h Filter sigma regulating filter strength for color.
    .   @param search_window Size of search window.
    .   @param block_size Size of block used for computing weights.
    .   @param borderMode Border type. See borderInterpolate for details. BORDER_REFLECT101 ,
    .   BORDER_REPLICATE , BORDER_CONSTANT , BORDER_REFLECT and BORDER_WRAP are supported for now.
    .   @param stream Stream for the asynchronous version.
    .   
    .   @sa
    .      fastNlMeansDenoising
    """
def printCudaDeviceInfo(device) -> None:
    """
    .
    """
def printShortCudaDeviceInfo(device) -> None:
    """
    .
    """
def registerPageLocked(m) -> None:
    """
    .   @brief Page-locks the memory of matrix and maps it for the device(s).
    .   
    .   @param m Input matrix.
    """
def resetDevice() -> None:
    """
    .   @brief Explicitly destroys and cleans up all resources associated with the current device in the current
    .   process.
    .   
    .   Any subsequent API call to this device will reinitialize the device.
    """
def setBufferPoolConfig(deviceId, stackSize, stackCount) -> None:
    """
    .
    """
def setBufferPoolUsage(on) -> None:
    """
    .
    """
def setDevice(device) -> None:
    """
    .   @brief Sets a device and initializes it for the current thread.
    .   
    .   @param device System index of a CUDA device starting with 0.
    .   
    .   If the call of this function is omitted, a default device is initialized at the fist CUDA usage.
    """
def unregisterPageLocked(m) -> None:
    """
    .   @brief Unmaps the memory of matrix and makes it pageable again.
    .   
    .   @param m Input matrix.
    """
DEVICE_INFO_COMPUTE_MODE_DEFAULT: int = 0
DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE: int = 1
DEVICE_INFO_COMPUTE_MODE_EXCLUSIVE_PROCESS: int = 3
DEVICE_INFO_COMPUTE_MODE_PROHIBITED: int = 2
DYNAMIC_PARALLELISM: int = 35
DeviceInfo_ComputeModeDefault: int = 0
DeviceInfo_ComputeModeExclusive: int = 1
DeviceInfo_ComputeModeExclusiveProcess: int = 3
DeviceInfo_ComputeModeProhibited: int = 2
EVENT_BLOCKING_SYNC: int = 1
EVENT_DEFAULT: int = 0
EVENT_DISABLE_TIMING: int = 2
EVENT_INTERPROCESS: int = 4
Event_BLOCKING_SYNC: int = 1
Event_DEFAULT: int = 0
Event_DISABLE_TIMING: int = 2
Event_INTERPROCESS: int = 4
FEATURE_SET_COMPUTE_10: int = 10
FEATURE_SET_COMPUTE_11: int = 11
FEATURE_SET_COMPUTE_12: int = 12
FEATURE_SET_COMPUTE_13: int = 13
FEATURE_SET_COMPUTE_20: int = 20
FEATURE_SET_COMPUTE_21: int = 21
FEATURE_SET_COMPUTE_30: int = 30
FEATURE_SET_COMPUTE_32: int = 32
FEATURE_SET_COMPUTE_35: int = 35
FEATURE_SET_COMPUTE_50: int = 50
GLOBAL_ATOMICS: int = 11
HOST_MEM_PAGE_LOCKED: int = 1
HOST_MEM_SHARED: int = 2
HOST_MEM_WRITE_COMBINED: int = 4
HostMem_PAGE_LOCKED: int = 1
HostMem_SHARED: int = 2
HostMem_WRITE_COMBINED: int = 4
NATIVE_DOUBLE: int = 13
SHARED_ATOMICS: int = 12
WARP_SHUFFLE_FUNCTIONS: int = 30
