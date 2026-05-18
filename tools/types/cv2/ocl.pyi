from __future__ import annotations
__all__: list[str] = ['DEVICE_EXEC_KERNEL', 'DEVICE_EXEC_NATIVE_KERNEL', 'DEVICE_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT', 'DEVICE_FP_DENORM', 'DEVICE_FP_FMA', 'DEVICE_FP_INF_NAN', 'DEVICE_FP_ROUND_TO_INF', 'DEVICE_FP_ROUND_TO_NEAREST', 'DEVICE_FP_ROUND_TO_ZERO', 'DEVICE_FP_SOFT_FLOAT', 'DEVICE_LOCAL_IS_GLOBAL', 'DEVICE_LOCAL_IS_LOCAL', 'DEVICE_NO_CACHE', 'DEVICE_NO_LOCAL_MEM', 'DEVICE_READ_ONLY_CACHE', 'DEVICE_READ_WRITE_CACHE', 'DEVICE_TYPE_ACCELERATOR', 'DEVICE_TYPE_ALL', 'DEVICE_TYPE_CPU', 'DEVICE_TYPE_DEFAULT', 'DEVICE_TYPE_DGPU', 'DEVICE_TYPE_GPU', 'DEVICE_TYPE_IGPU', 'DEVICE_UNKNOWN_VENDOR', 'DEVICE_VENDOR_AMD', 'DEVICE_VENDOR_INTEL', 'DEVICE_VENDOR_NVIDIA', 'Device', 'Device_EXEC_KERNEL', 'Device_EXEC_NATIVE_KERNEL', 'Device_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT', 'Device_FP_DENORM', 'Device_FP_FMA', 'Device_FP_INF_NAN', 'Device_FP_ROUND_TO_INF', 'Device_FP_ROUND_TO_NEAREST', 'Device_FP_ROUND_TO_ZERO', 'Device_FP_SOFT_FLOAT', 'Device_LOCAL_IS_GLOBAL', 'Device_LOCAL_IS_LOCAL', 'Device_NO_CACHE', 'Device_NO_LOCAL_MEM', 'Device_READ_ONLY_CACHE', 'Device_READ_WRITE_CACHE', 'Device_TYPE_ACCELERATOR', 'Device_TYPE_ALL', 'Device_TYPE_CPU', 'Device_TYPE_DEFAULT', 'Device_TYPE_DGPU', 'Device_TYPE_GPU', 'Device_TYPE_IGPU', 'Device_UNKNOWN_VENDOR', 'Device_VENDOR_AMD', 'Device_VENDOR_INTEL', 'Device_VENDOR_NVIDIA', 'Device_getDefault', 'KERNEL_ARG_CONSTANT', 'KERNEL_ARG_LOCAL', 'KERNEL_ARG_NO_SIZE', 'KERNEL_ARG_PTR_ONLY', 'KERNEL_ARG_READ_ONLY', 'KERNEL_ARG_READ_WRITE', 'KERNEL_ARG_WRITE_ONLY', 'KernelArg_CONSTANT', 'KernelArg_LOCAL', 'KernelArg_NO_SIZE', 'KernelArg_PTR_ONLY', 'KernelArg_READ_ONLY', 'KernelArg_READ_WRITE', 'KernelArg_WRITE_ONLY', 'OCL_VECTOR_DEFAULT', 'OCL_VECTOR_MAX', 'OCL_VECTOR_OWN', 'OpenCLExecutionContext', 'finish', 'haveAmdBlas', 'haveAmdFft', 'haveOpenCL', 'setUseOpenCL', 'useOpenCL']
class Device:
    @staticmethod
    def OpenCLVersion() -> retval:
        """
        .
        """
    @staticmethod
    def OpenCL_C_Version() -> retval:
        """
        .
        """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def addressBits() -> retval:
        """
        .
        """
    @staticmethod
    def available() -> retval:
        """
        .
        """
    @staticmethod
    def compilerAvailable() -> retval:
        """
        .
        """
    @staticmethod
    def deviceVersionMajor() -> retval:
        """
        .
        """
    @staticmethod
    def deviceVersionMinor() -> retval:
        """
        .
        """
    @staticmethod
    def doubleFPConfig() -> retval:
        """
        .
        """
    @staticmethod
    def driverVersion() -> retval:
        """
        .
        """
    @staticmethod
    def endianLittle() -> retval:
        """
        .
        """
    @staticmethod
    def errorCorrectionSupport() -> retval:
        """
        .
        """
    @staticmethod
    def executionCapabilities() -> retval:
        """
        .
        """
    @staticmethod
    def extensions() -> retval:
        """
        .
        """
    @staticmethod
    def getDefault() -> retval:
        """
        .
        """
    @staticmethod
    def globalMemCacheLineSize() -> retval:
        """
        .
        """
    @staticmethod
    def globalMemCacheSize() -> retval:
        """
        .
        """
    @staticmethod
    def globalMemCacheType() -> retval:
        """
        .
        """
    @staticmethod
    def globalMemSize() -> retval:
        """
        .
        """
    @staticmethod
    def halfFPConfig() -> retval:
        """
        .
        """
    @staticmethod
    def hostUnifiedMemory() -> retval:
        """
        .
        """
    @staticmethod
    def image2DMaxHeight() -> retval:
        """
        .
        """
    @staticmethod
    def image2DMaxWidth() -> retval:
        """
        .
        """
    @staticmethod
    def image3DMaxDepth() -> retval:
        """
        .
        """
    @staticmethod
    def image3DMaxHeight() -> retval:
        """
        .
        """
    @staticmethod
    def image3DMaxWidth() -> retval:
        """
        .
        """
    @staticmethod
    def imageFromBufferSupport() -> retval:
        """
        .
        """
    @staticmethod
    def imageMaxArraySize() -> retval:
        """
        .
        """
    @staticmethod
    def imageMaxBufferSize() -> retval:
        """
        .
        """
    @staticmethod
    def imageSupport() -> retval:
        """
        .
        """
    @staticmethod
    def intelSubgroupsSupport() -> retval:
        """
        .
        """
    @staticmethod
    def isAMD() -> retval:
        """
        .
        """
    @staticmethod
    def isExtensionSupported(extensionName) -> retval:
        """
        .
        """
    @staticmethod
    def isIntel() -> retval:
        """
        .
        """
    @staticmethod
    def isNVidia() -> retval:
        """
        .
        """
    @staticmethod
    def linkerAvailable() -> retval:
        """
        .
        """
    @staticmethod
    def localMemSize() -> retval:
        """
        .
        """
    @staticmethod
    def localMemType() -> retval:
        """
        .
        """
    @staticmethod
    def maxClockFrequency() -> retval:
        """
        .
        """
    @staticmethod
    def maxComputeUnits() -> retval:
        """
        .
        """
    @staticmethod
    def maxConstantArgs() -> retval:
        """
        .
        """
    @staticmethod
    def maxConstantBufferSize() -> retval:
        """
        .
        """
    @staticmethod
    def maxMemAllocSize() -> retval:
        """
        .
        """
    @staticmethod
    def maxParameterSize() -> retval:
        """
        .
        """
    @staticmethod
    def maxReadImageArgs() -> retval:
        """
        .
        """
    @staticmethod
    def maxSamplers() -> retval:
        """
        .
        """
    @staticmethod
    def maxWorkGroupSize() -> retval:
        """
        .
        """
    @staticmethod
    def maxWorkItemDims() -> retval:
        """
        .
        """
    @staticmethod
    def maxWriteImageArgs() -> retval:
        """
        .
        """
    @staticmethod
    def memBaseAddrAlign() -> retval:
        """
        .
        """
    @staticmethod
    def name() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthChar() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthDouble() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthFloat() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthHalf() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthInt() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthLong() -> retval:
        """
        .
        """
    @staticmethod
    def nativeVectorWidthShort() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthChar() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthDouble() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthFloat() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthHalf() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthInt() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthLong() -> retval:
        """
        .
        """
    @staticmethod
    def preferredVectorWidthShort() -> retval:
        """
        .
        """
    @staticmethod
    def printfBufferSize() -> retval:
        """
        .
        """
    @staticmethod
    def profilingTimerResolution() -> retval:
        """
        .
        """
    @staticmethod
    def singleFPConfig() -> retval:
        """
        .
        """
    @staticmethod
    def type() -> retval:
        """
        .
        """
    @staticmethod
    def vendorID() -> retval:
        """
        .
        """
    @staticmethod
    def vendorName() -> retval:
        """
        .
        """
    @staticmethod
    def version() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class OpenCLExecutionContext:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def Device_getDefault() -> retval:
    """
    .
    """
def finish() -> None:
    """
    .
    """
def haveAmdBlas() -> retval:
    """
    .
    """
def haveAmdFft() -> retval:
    """
    .
    """
def haveOpenCL() -> retval:
    """
    .
    """
def setUseOpenCL(flag) -> None:
    """
    .
    """
def useOpenCL() -> retval:
    """
    .
    """
DEVICE_EXEC_KERNEL: int = 1
DEVICE_EXEC_NATIVE_KERNEL: int = 2
DEVICE_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: int = 128
DEVICE_FP_DENORM: int = 1
DEVICE_FP_FMA: int = 32
DEVICE_FP_INF_NAN: int = 2
DEVICE_FP_ROUND_TO_INF: int = 16
DEVICE_FP_ROUND_TO_NEAREST: int = 4
DEVICE_FP_ROUND_TO_ZERO: int = 8
DEVICE_FP_SOFT_FLOAT: int = 64
DEVICE_LOCAL_IS_GLOBAL: int = 2
DEVICE_LOCAL_IS_LOCAL: int = 1
DEVICE_NO_CACHE: int = 0
DEVICE_NO_LOCAL_MEM: int = 0
DEVICE_READ_ONLY_CACHE: int = 1
DEVICE_READ_WRITE_CACHE: int = 2
DEVICE_TYPE_ACCELERATOR: int = 8
DEVICE_TYPE_ALL: int = 4294967295
DEVICE_TYPE_CPU: int = 2
DEVICE_TYPE_DEFAULT: int = 1
DEVICE_TYPE_DGPU: int = 65540
DEVICE_TYPE_GPU: int = 4
DEVICE_TYPE_IGPU: int = 131076
DEVICE_UNKNOWN_VENDOR: int = 0
DEVICE_VENDOR_AMD: int = 1
DEVICE_VENDOR_INTEL: int = 2
DEVICE_VENDOR_NVIDIA: int = 3
Device_EXEC_KERNEL: int = 1
Device_EXEC_NATIVE_KERNEL: int = 2
Device_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: int = 128
Device_FP_DENORM: int = 1
Device_FP_FMA: int = 32
Device_FP_INF_NAN: int = 2
Device_FP_ROUND_TO_INF: int = 16
Device_FP_ROUND_TO_NEAREST: int = 4
Device_FP_ROUND_TO_ZERO: int = 8
Device_FP_SOFT_FLOAT: int = 64
Device_LOCAL_IS_GLOBAL: int = 2
Device_LOCAL_IS_LOCAL: int = 1
Device_NO_CACHE: int = 0
Device_NO_LOCAL_MEM: int = 0
Device_READ_ONLY_CACHE: int = 1
Device_READ_WRITE_CACHE: int = 2
Device_TYPE_ACCELERATOR: int = 8
Device_TYPE_ALL: int = 4294967295
Device_TYPE_CPU: int = 2
Device_TYPE_DEFAULT: int = 1
Device_TYPE_DGPU: int = 65540
Device_TYPE_GPU: int = 4
Device_TYPE_IGPU: int = 131076
Device_UNKNOWN_VENDOR: int = 0
Device_VENDOR_AMD: int = 1
Device_VENDOR_INTEL: int = 2
Device_VENDOR_NVIDIA: int = 3
KERNEL_ARG_CONSTANT: int = 8
KERNEL_ARG_LOCAL: int = 1
KERNEL_ARG_NO_SIZE: int = 256
KERNEL_ARG_PTR_ONLY: int = 16
KERNEL_ARG_READ_ONLY: int = 2
KERNEL_ARG_READ_WRITE: int = 6
KERNEL_ARG_WRITE_ONLY: int = 4
KernelArg_CONSTANT: int = 8
KernelArg_LOCAL: int = 1
KernelArg_NO_SIZE: int = 256
KernelArg_PTR_ONLY: int = 16
KernelArg_READ_ONLY: int = 2
KernelArg_READ_WRITE: int = 6
KernelArg_WRITE_ONLY: int = 4
OCL_VECTOR_DEFAULT: int = 0
OCL_VECTOR_MAX: int = 1
OCL_VECTOR_OWN: int = 0
