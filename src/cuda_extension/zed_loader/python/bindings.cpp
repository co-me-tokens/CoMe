#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "zed_loader/common.hpp"
#include "zed_loader/zed_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(zed_loader, m) {
    m.doc() = "ZED Camera loader with PyTorch tensor support";
    
    // Resolution enum
    py::enum_<zed_loader::Resolution>(m, "Resolution",
        "Camera resolution options for ZED cameras")
        .value("HD2K", zed_loader::Resolution::HD2K, "2208x1242, 15fps")
        .value("HD1080", zed_loader::Resolution::HD1080, "1920x1080, 30fps")
        .value("HD1200", zed_loader::Resolution::HD1200, "1920x1200, 60fps (ZED-X only)")
        .value("HD720", zed_loader::Resolution::HD720, "1280x720, 60fps")
        .value("SVGA", zed_loader::Resolution::SVGA, "960x600, 120fps (ZED-X only)")
        .value("VGA", zed_loader::Resolution::VGA, "672x376, 100fps")
        .value("AUTO", zed_loader::Resolution::AUTO, "Auto select based on camera")
        .export_values();

    // SensorParameters struct
    py::class_<zed_loader::SensorParameters>(m, "SensorParameters",
        "IMU sensor parameters (accelerometer or gyroscope)")
        .def_readonly("sampling_rate", &zed_loader::SensorParameters::sampling_rate,
            "Sampling rate in Hz")
        .def_readonly("range_min", &zed_loader::SensorParameters::range_min,
            "Minimum measurement range")
        .def_readonly("range_max", &zed_loader::SensorParameters::range_max,
            "Maximum measurement range")
        .def_readonly("resolution", &zed_loader::SensorParameters::resolution,
            "Sensor resolution")
        .def_readonly("noise_density", &zed_loader::SensorParameters::noise_density,
            "Noise density")
        .def_readonly("random_walk", &zed_loader::SensorParameters::random_walk,
            "Random walk")
        .def("__repr__", [](const zed_loader::SensorParameters& p) {
            return "<SensorParameters sampling_rate=" + std::to_string(p.sampling_rate) +
                   " range=[" + std::to_string(p.range_min) + ", " + std::to_string(p.range_max) + "]" +
                   " resolution=" + std::to_string(p.resolution) +
                   " noise_density=" + std::to_string(p.noise_density) +
                   " random_walk=" + std::to_string(p.random_walk) + ">";
        });

    // Register custom exceptions
    // Base exception inherits from Python's RuntimeError
    static py::exception<zed_loader::ZedLoaderError> exc_base(m, "ZedLoaderError");
    
    // Derived exceptions inherit from base
    static py::exception<zed_loader::CameraNotFoundError> exc_not_found(
        m, "CameraNotFoundError", exc_base.ptr());
    static py::exception<zed_loader::CameraOpenError> exc_open(
        m, "CameraOpenError", exc_base.ptr());
    static py::exception<zed_loader::InvalidParameterError> exc_param(
        m, "InvalidParameterError", exc_base.ptr());
    static py::exception<zed_loader::InternalError> exc_internal(
            m, "InternalError", exc_base.ptr());

    // Register exception translators (for automatic C++ -> Python translation)
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const zed_loader::CameraNotFoundError& e) {
            py::set_error(exc_not_found, e.what());
        } catch (const zed_loader::CameraOpenError& e) {
            py::set_error(exc_open, e.what());
        } catch (const zed_loader::InvalidParameterError& e) {
            py::set_error(exc_param, e.what());
        } catch (const zed_loader::InternalError& e) {
            py::set_error(exc_internal, e.what());
        } catch (const zed_loader::ZedLoaderError& e) {
            py::set_error(exc_base, e.what());
        }
    });
    
    // ZedDataLoader class binding
    py::class_<zed_loader::ZedDataLoader>(m, "ZedDataLoader",
        R"doc(
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
        )doc")
        .def(py::init<zed_loader::Resolution, int, int, int>(),
             py::arg("resolution") = zed_loader::Resolution::HD720,
             py::arg("exposure") = -1,
             py::arg("gain") = -1,
             py::arg("fps") = -1,
             R"doc(
             Create a ZedDataLoader with specified settings.
             
             Args:
                 resolution: Camera resolution (default: Resolution.HD720)
                 exposure: Exposure time in microseconds. -1 for auto (default).
                 gain: Sensor gain. -1 for auto (default). Valid range: 0-100.
                 fps: Target frames per second. -1 for maximum FPS (default).
             )doc")
        .def("start", &zed_loader::ZedDataLoader::start,
             py::call_guard<py::gil_scoped_release>(),
             R"doc(
             Start the worker threads for IMU and image capture.
             
             Raises:
                 CameraOpenError: If camera fails to open
             )doc")
        .def("stop", &zed_loader::ZedDataLoader::stop,
             py::call_guard<py::gil_scoped_release>(),
             "Stop the worker threads and close the camera.")
        .def("is_running", &zed_loader::ZedDataLoader::is_running,
             "Check if the data loader is currently running.")
        .def("get_data", [](zed_loader::ZedDataLoader& self) {
                // Get data with GIL released
                std::optional<zed_loader::ZedDataLoader::IMUDataResult> imu_opt;
                std::optional<zed_loader::ZedDataLoader::ImageDataResult> image_opt;
                {
                    py::gil_scoped_release release;
                    auto result = self.get_data();
                    imu_opt = std::move(std::get<0>(result));
                    image_opt = std::move(std::get<1>(result));
                }

                // Convert C++ optionals to Python None or tuples (GIL is held here)
                py::object imu_result = py::none();
                py::object image_result = py::none();

                if (imu_opt.has_value()) {
                    auto& imu_tuple = *imu_opt;
                    imu_result = py::make_tuple(
                        std::get<0>(imu_tuple),
                        std::get<1>(imu_tuple),
                        std::get<2>(imu_tuple)
                    );
                }

                if (image_opt.has_value()) {
                    auto& image_tuple = *image_opt;
                    image_result = py::make_tuple(
                        std::get<0>(image_tuple),
                        py::int_(std::get<1>(image_tuple))
                    );
                }

                return py::make_tuple(imu_result, image_result);
            },
            R"doc(
            Get the latest IMU and image data.

            Returns:
                tuple: (imu_data, image_data) where:
                    - imu_data: tuple(accel[N,3], gyro[N,3], timestamps[N]) or None
                    - image_data: tuple(stereo[2,H,W,C], timestamp_ns) or None
            )doc")
        .def_property_readonly("left_intrinsics", 
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.left_intrinsics();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Left camera intrinsic matrix (3x3).
            
            Returns:
                torch.Tensor: Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                              or None if not started.
            )doc")
        .def_property_readonly("right_intrinsics",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.right_intrinsics();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Right camera intrinsic matrix (3x3).
            
            Returns:
                torch.Tensor: Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                              or None if not started.
            )doc")
        .def_property_readonly("left_distortion",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.left_distortion();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Left camera distortion coefficients.
            
            Returns:
                torch.Tensor: Distortion coefficients [k1, k2, p1, p2, k3]
                              or None if not started.
            )doc")
        .def_property_readonly("right_distortion",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.right_distortion();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Right camera distortion coefficients.
            
            Returns:
                torch.Tensor: Distortion coefficients [k1, k2, p1, p2, k3]
                              or None if not started.
            )doc")
        .def_property_readonly("baseline",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.baseline();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Stereo camera baseline in meters.
            
            Returns:
                float: Baseline distance in meters, or None if not started.
            )doc")
        .def_property_readonly("accelerometer_params",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.accelerometer_params();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Accelerometer sensor parameters.
            
            Returns:
                SensorParameters: Accelerometer parameters including sampling_rate,
                                  range, resolution, noise_density, random_walk.
                                  Returns None if not started.
            )doc")
        .def_property_readonly("gyroscope_params",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.gyroscope_params();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            Gyroscope sensor parameters.
            
            Returns:
                SensorParameters: Gyroscope parameters including sampling_rate,
                                  range, resolution, noise_density, random_walk.
                                  Returns None if not started.
            )doc")
        .def_property_readonly("left_T_BS",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.left_T_BS();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            T_BS for the left camera (4x4): Body <- Sensor.
            
            Returns the transformation matrix that transforms points FROM the
            left camera frame TO the body frame (IMU).
            
            Convention: T_BS means "Body <- Sensor" (sensor-to-body transform).
            
            Returns:
                torch.Tensor: 4x4 homogeneous transformation matrix,
                              or None if not started.
            )doc")
        .def_property_readonly("right_T_BS",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.right_T_BS();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            T_BS for the right camera (4x4): Body <- Sensor.
            
            Returns the transformation matrix that transforms points FROM the
            right camera frame TO the body frame (IMU).
            
            Convention: T_BS means "Body <- Sensor" (sensor-to-body transform).
            
            Returns:
                torch.Tensor: 4x4 homogeneous transformation matrix,
                              or None if not started.
            )doc")
        .def_property_readonly("imu_T_BS",
            [](zed_loader::ZedDataLoader& self) -> py::object {
                auto opt = self.imu_T_BS();
                if (opt.has_value()) {
                    return py::cast(*opt);
                }
                return py::none();
            },
            R"doc(
            T_BS for the IMU (4x4): Body <- Sensor.
            
            Since the IMU is the body frame, this is the identity matrix.
            
            Returns:
                torch.Tensor: 4x4 identity matrix, or None if not started.
            )doc")
        .def("__enter__", [](zed_loader::ZedDataLoader& self) -> zed_loader::ZedDataLoader& {
            self.start();
            return self;
        })
        .def("__exit__", [](zed_loader::ZedDataLoader& self, 
                           [[maybe_unused]] py::object exc_type,
                           [[maybe_unused]] py::object exc_val,
                           [[maybe_unused]] py::object exc_tb) {
            self.stop();
            return false;  // Don't suppress exceptions
        });
}
