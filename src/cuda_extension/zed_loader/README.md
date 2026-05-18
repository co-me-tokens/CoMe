# ZedLoader

High-performance C++ ZED camera data loader with Python bindings, PyTorch/ATen tensor integration, and multi-threaded capture.

## Features

- **Multi-threaded Capture**: Separate worker threads for IMU and image acquisition
- **Double-buffering**: Lock-free producer/consumer pattern prevents data drops
- **PyTorch Integration**: Direct conversion to `torch.Tensor` for zero-copy ML workflows
- **Stereo + IMU**: Captures synchronized stereo images and high-rate IMU data
- **Camera Calibration**: Provides intrinsic matrices, distortion coefficients, and stereo baseline
- **Context Manager**: Pythonic `with` statement support for automatic resource cleanup
- **Type Hints**: Full `.pyi` stub file for IDE autocompletion

## Requirements

| Dependency | Version |
|------------|---------|
| ZED SDK | 4.x |
| PyTorch | ≥ 2.0 |
| CUDA Toolkit | (matching ZED SDK) |
| CMake | ≥ 3.18 |
| C++ Compiler | C++17 compatible |
| pybind11 | ≥ 2.10 |
| Python | ≥ 3.8 |

## Installation

### Option 1: pip install (Recommended)

```bash
cd Src/CUExt/ZedLoader
pip install -e .
```

### Option 2: Build Script

```bash
cd Src/CUExt/ZedLoader
./install.sh              # Full build (core + python + demo executable)
./install.sh --no-exec    # Without demo executable (no Rerun SDK needed)
```

### Option 3: Manual CMake Build

```bash
cd Src/CUExt/ZedLoader
mkdir build && cd build
cmake -DBUILD_ZEDLOADER_EXEC=OFF ..
make -j$(nproc)

# The Python module is built in the source directory
```

## Quick Start

```python
from zed_loader import ZedDataLoader, Resolution

# Using context manager (recommended)
with ZedDataLoader(Resolution.HD720, fps=30) as loader:
    imu_data, image_data = loader.get_data()
    
    if imu_data is not None:
        accel, gyro, timestamps = imu_data
        print(f"IMU: {accel.shape[0]} samples")
    
    if image_data is not None:
        stereo_image, timestamp_ns = image_data
        print(f"Image: {stereo_image.shape}")  # [2, H, W, 4] (BGRA)

# Or manual start/stop
loader = ZedDataLoader(Resolution.HD1080, exposure=5000, gain=50)
loader.start()
# ... capture loop ...
loader.stop()
```

## API Reference

### `ZedDataLoader`

```python
ZedDataLoader(
    resolution: Resolution = Resolution.HD720,
    exposure: int = -1,      # -1 for auto, or 28-30000 microseconds
    gain: int = -1,          # -1 for auto, or 0-100
    fps: int = -1            # -1 for max rate, or target FPS
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Open camera and start worker threads |
| `stop()` | Stop workers and close camera |
| `is_running()` | Check if capture is active |
| `get_data()` | Get latest buffered IMU and image data |

#### Properties (available after `start()`)

| Property | Type | Description |
|----------|------|-------------|
| `left_intrinsics` | `Tensor[3,3]` | Left camera matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `right_intrinsics` | `Tensor[3,3]` | Right camera matrix |
| `left_distortion` | `Tensor[5]` | Left distortion `[k1,k2,p1,p2,k3]` |
| `right_distortion` | `Tensor[5]` | Right distortion coefficients |
| `baseline` | `float` | Stereo baseline in meters |
| `accelerometer_params` | `SensorParameters` | Accelerometer specs |
| `gyroscope_params` | `SensorParameters` | Gyroscope specs |
| `left_T_BS` | `Tensor[4,4]` | Left camera → Body (IMU) transform |
| `right_T_BS` | `Tensor[4,4]` | Right camera → Body (IMU) transform |
| `imu_T_BS` | `Tensor[4,4]` | IMU → Body transform (identity) |

### `Resolution` Enum

| Value | Resolution | Max FPS | Notes |
|-------|------------|---------|-------|
| `HD2K` | 2208×1242 | 15 | |
| `HD1080` | 1920×1080 | 30 | |
| `HD1200` | 1920×1200 | 60 | ZED-X only |
| `HD720` | 1280×720 | 60 | Default |
| `SVGA` | 960×600 | 120 | ZED-X only |
| `VGA` | 672×376 | 100 | |
| `AUTO` | Auto | - | Camera-dependent |

### `SensorParameters`

IMU sensor characteristics (accelerometer or gyroscope):

| Attribute | Description |
|-----------|-------------|
| `sampling_rate` | Sampling rate in Hz |
| `range_min` / `range_max` | Measurement range |
| `resolution` | Sensor resolution |
| `noise_density` | Noise density |
| `random_walk` | Random walk |

### Sensor-to-Body Transforms (T_BS)

The `T_BS` properties provide 4×4 homogeneous transformation matrices following the **"Body ← Sensor"** convention (transforms points FROM sensor TO body frame):

```python
with ZedDataLoader() as loader:
    # Get transforms after camera is started
    T_left = loader.left_T_BS    # Left Camera → IMU (Body)
    T_right = loader.right_T_BS  # Right Camera → IMU (Body)
    T_imu = loader.imu_T_BS      # IMU → IMU (identity)
    
    # Transform a point from left camera frame to body (IMU) frame
    point_cam = torch.tensor([x, y, z, 1.0])
    point_body = T_left @ point_cam
```

| Property | Description |
|----------|-------------|
| `left_T_BS` | Transform: Left Camera → Body (IMU) |
| `right_T_BS` | Transform: Right Camera → Body (IMU) |
| `imu_T_BS` | Identity matrix (IMU is the body frame) |

**Convention**: `T_BS` means "Body ← Sensor" — the matrix transforms points FROM the sensor frame TO the body frame. This is consistent with the codebase convention.

These transforms are computed from ZED SDK's factory calibration by inverting `sensors_configuration.camera_imu_transform` and `calibration_parameters.stereo_transform`.

### Data Formats

#### `get_data()` Return Value

```python
imu_data, image_data = loader.get_data()

# imu_data: tuple or None
#   accel: Tensor[N, 3]     - Linear acceleration [m/s²]
#   gyro:  Tensor[N, 3]     - Angular velocity [deg/s]
#   timestamps: Tensor[N]   - Timestamps [nanoseconds, int64]

# image_data: tuple or None
#   stereo: Tensor[2, H, W, 4]  - Stereo BGRA images (uint8)
#   timestamp_ns: int            - Image timestamp [nanoseconds]
```

## Exception Handling

```python
from zed_loader import (
    ZedLoaderError,       # Base exception
    CameraNotFoundError,  # No ZED camera detected
    CameraOpenError,      # Failed to open camera
    InvalidParameterError # Invalid configuration
)

try:
    with ZedDataLoader() as loader:
        ...
except CameraNotFoundError:
    print("Please connect a ZED camera")
except CameraOpenError as e:
    print(f"Camera error: {e}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ZedDataLoader                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │  IMU Worker     │      │  Image Worker                │  │
│  │  Thread         │      │  Thread                      │  │
│  │                 │      │                              │  │
│  │  polls @ 400Hz  │      │  grabs frames @ target fps   │  │
│  └────────┬────────┘      └──────────────┬───────────────┘  │
│           │                              │                  │
│           ▼                              ▼                  │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │  Double Buffer  │      │  Double Buffer               │  │
│  │  (IMU Ring)     │      │  (Stereo Image)              │  │
│  └────────┬────────┘      └──────────────┬───────────────┘  │
│           │                              │                  │
│           └──────────────┬───────────────┘                  │
│                          ▼                                  │
│                   ┌─────────────┐                           │
│                   │  get_data() │  ← User calls from Python │
│                   │  → Tensors  │                           │
│                   └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ZedLoader/
├── CMakeLists.txt           # CMake build configuration
├── pyproject.toml           # Python package metadata
├── setup.py                 # pip install support
├── install.sh               # Build script
├── __init__.py              # Python package init
├── zed_loader.pyi           # Type stubs for IDE support
├── include/
│   └── zed_loader/
│       ├── common.hpp       # Types, buffers, exceptions
│       └── zed_loader.hpp   # ZedDataLoader class
├── src/
│   ├── zed_loader.cpp       # Core implementation
│   └── main.cpp             # Demo executable (Rerun visualization)
└── python/
    └── bindings.cpp         # pybind11 Python bindings
```

## Demo Executable

The optional demo executable visualizes camera data using [Rerun](https://rerun.io/):

```bash
# Build with executable (requires Rerun SDK at /usr/local/rerun_cpp_sdk)
./install.sh

# Start Rerun viewer, then run the demo
cd build
./zed_loader_exec
```

## License

MIT License - see [pyproject.toml](pyproject.toml) for details.
