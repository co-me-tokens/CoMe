"""
ZED Camera loader with PyTorch tensor support
"""
from __future__ import annotations
import typing
import enum
import torch

# Type aliases for clarity
IMUData = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (accel[N,3], gyro[N,3], timestamps[N])
ImageData = tuple[torch.Tensor, int]  # (stereo[2,H,W,C], timestamp_ns)

__all__: list[str] = [
    'Resolution',
    'SensorParameters',
    'CameraNotFoundError', 
    'CameraOpenError', 
    'InternalError', 
    'InvalidParameterError', 
    'ZedDataLoader', 
    'ZedLoaderError'
]

class Resolution(enum.IntEnum):
    """Camera resolution options for ZED cameras."""
    HD2K = 0    # 2208x1242, 15fps
    HD1080 = 1  # 1920x1080, 30fps
    HD1200 = 2  # 1920x1200, 60fps (ZED-X only)
    HD720 = 3   # 1280x720, 60fps
    SVGA = 4    # 960x600, 120fps (ZED-X only)
    VGA = 5     # 672x376, 100fps
    AUTO = 6    # Auto select based on camera

class SensorParameters:
    """IMU sensor parameters (accelerometer or gyroscope)."""
    sampling_rate: float   # Sampling rate in Hz
    range_min: float       # Minimum measurement range
    range_max: float       # Maximum measurement range
    resolution: float      # Sensor resolution
    noise_density: float   # Noise density
    random_walk: float     # Random walk

class ZedLoaderError(Exception):
    """Base exception for ZedLoader errors."""
    pass
class CameraNotFoundError(ZedLoaderError):
    pass
class CameraOpenError(ZedLoaderError):
    pass
class InternalError(ZedLoaderError):
    pass
class InvalidParameterError(ZedLoaderError):
    pass
class ZedDataLoader:
    """
    Multi-threaded ZED camera data loader.

    This class provides concurrent access to ZED camera IMU and image data
    using a double-buffering strategy to avoid data drops.

    Args:
        resolution: Camera resolution (default: Resolution.HD720)
        exposure: Exposure time in microseconds. -1 for auto (default).
                 Valid range depends on resolution/fps, typically 28-30000.
        gain: Sensor gain. -1 for auto (default). Valid range: 0-100.
        fps: Target frames per second. -1 for maximum FPS (default).
             If positive, the image worker will sleep to maintain this rate.

    Example:
        >>> loader = ZedDataLoader(Resolution.HD1080, exposure=5000, gain=50, fps=30)
        >>> loader.start()
        >>> imu_data, image_data = loader.get_data()
        >>> if imu_data is not None:
        ...     accel, gyro, timestamps = imu_data
        >>> if image_data is not None:
        ...     stereo_image, timestamp_ns = image_data
        >>> loader.stop()
    """
    def __enter__(self) -> ZedDataLoader:
        ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any) -> bool:
        ...
    def __init__(
        self,
        resolution: Resolution = Resolution.HD720,
        exposure: int = -1,
        gain: int = -1,
        fps: int = -1
    ) -> None:
        """
        Create a ZedDataLoader with specified settings.
        
        Args:
            resolution: Camera resolution (default: Resolution.HD720)
            exposure: Exposure time in microseconds. -1 for auto (default).
            gain: Sensor gain. -1 for auto (default). Valid range: 0-100.
            fps: Target frames per second. -1 for maximum FPS (default).
        """
    def get_data(self) -> tuple[IMUData | None, ImageData | None]:
        """
        Get the latest IMU and image data.
        
        Returns:
            tuple: (imu_data, image_data) where:
                - imu_data: tuple(accel[N,3], gyro[N,3], timestamps[N]) or None
                - image_data: tuple(stereo[2,H,W,C], timestamp_ns) or None
        """
    def is_running(self) -> bool:
        """
        Check if the data loader is currently running.
        """
    def start(self) -> None:
        """
        Start the worker threads for IMU and image capture.
        
        Raises:
            CameraOpenError: If camera fails to open
        """
    def stop(self) -> None:
        """
        Stop the worker threads and close the camera.
        """
    
    @property
    def left_intrinsics(self) -> torch.Tensor | None:
        """
        Left camera intrinsic matrix (3x3).
        
        Returns:
            Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or None if not started.
        """
    
    @property
    def right_intrinsics(self) -> torch.Tensor | None:
        """
        Right camera intrinsic matrix (3x3).
        
        Returns:
            Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or None if not started.
        """
    
    @property
    def left_distortion(self) -> torch.Tensor | None:
        """
        Left camera distortion coefficients.
        
        Returns:
            Distortion coefficients [k1, k2, p1, p2, k3] or None if not started.
        """
    
    @property
    def right_distortion(self) -> torch.Tensor | None:
        """
        Right camera distortion coefficients.
        
        Returns:
            Distortion coefficients [k1, k2, p1, p2, k3] or None if not started.
        """
    
    @property
    def baseline(self) -> float | None:
        """
        Stereo camera baseline in meters.
        
        Returns:
            Baseline distance in meters, or None if not started.
        """
    
    @property
    def accelerometer_params(self) -> SensorParameters | None:
        """
        Accelerometer sensor parameters.
        
        Returns:
            SensorParameters with sampling_rate, range, resolution, noise_density, 
            random_walk. Returns None if not started.
        """
    
    @property
    def gyroscope_params(self) -> SensorParameters | None:
        """
        Gyroscope sensor parameters.
        
        Returns:
            SensorParameters with sampling_rate, range, resolution, noise_density, 
            random_walk. Returns None if not started.
        """
    
    @property
    def left_T_BS(self) -> torch.Tensor | None:
        """
        T_BS for the left camera (4x4): Body <- Sensor.
        
        Returns the transformation matrix that transforms points FROM the
        left camera frame TO the body frame (IMU).
        
        Convention: T_BS means "Body <- Sensor" (sensor-to-body transform).
        
        Returns:
            4x4 homogeneous transformation matrix, or None if not started.
        """
    
    @property
    def right_T_BS(self) -> torch.Tensor | None:
        """
        T_BS for the right camera (4x4): Body <- Sensor.
        
        Returns the transformation matrix that transforms points FROM the
        right camera frame TO the body frame (IMU).
        
        Convention: T_BS means "Body <- Sensor" (sensor-to-body transform).
        
        Returns:
            4x4 homogeneous transformation matrix, or None if not started.
        """
    
    @property
    def imu_T_BS(self) -> torch.Tensor | None:
        """
        T_BS for the IMU (4x4): Body <- Sensor.
        
        Since the IMU is the body frame, this is the identity matrix.
        
        Returns:
            4x4 identity matrix, or None if not started.
        """