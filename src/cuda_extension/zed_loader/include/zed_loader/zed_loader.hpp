#pragma once

#include "zed_loader/common.hpp"

#include <cstddef>
#include <tuple>

namespace zed_loader {

    // Forward declarations for free functions
    bool reportCameraStatus(sl::Camera& camera);
    bool retrieve_latest_imu_reading(sl::Camera& zed_camera, IMU_Measurement& measurement);

    /**
     * @brief Camera resolution options (mirrors sl::RESOLUTION for Python interop)
     */
    enum class Resolution {
        HD2K = 0,   ///< 2208x1242, 15fps
        HD1080 = 1, ///< 1920x1080, 30fps
        HD1200 = 2, ///< 1920x1200, 60fps (ZED-X only)
        HD720 = 3,  ///< 1280x720, 60fps
        SVGA = 4,   ///< 960x600, 120fps (ZED-X only)
        VGA = 5,    ///< 672x376, 100fps
        AUTO = 6    ///< Auto select based on camera
    };

    /// Convert our Resolution enum to ZED SDK's sl::RESOLUTION
    inline sl::RESOLUTION to_sl_resolution(Resolution res) {
        switch (res) {
            case Resolution::HD2K:   return sl::RESOLUTION::HD2K;
            case Resolution::HD1080: return sl::RESOLUTION::HD1080;
            case Resolution::HD1200: return sl::RESOLUTION::HD1200;
            case Resolution::HD720:  return sl::RESOLUTION::HD720;
            case Resolution::SVGA:   return sl::RESOLUTION::SVGA;
            case Resolution::VGA:    return sl::RESOLUTION::VGA;
            case Resolution::AUTO:   return sl::RESOLUTION::AUTO;
            default:                 return sl::RESOLUTION::HD720;
        }
    }

    /**
     * @brief IMU sensor parameters (accelerometer or gyroscope).
     */
    struct SensorParameters {
        float sampling_rate;   ///< Sampling rate in Hz
        float range_min;       ///< Minimum measurement range
        float range_max;       ///< Maximum measurement range
        float resolution;      ///< Sensor resolution
        float noise_density;   ///< Noise density
        float random_walk;     ///< Random walk

        /// Convert from ZED SDK SensorParameters
        static SensorParameters from_sl(const sl::SensorParameters& params) {
            return SensorParameters{
                params.sampling_rate,
                params.range.x,
                params.range.y,
                params.resolution,
                params.noise_density,
                params.random_walk
            };
        }
    };

    /**
     * @brief Multi-threaded ZED camera data loader.
     * 
     * This class provides concurrent access to ZED camera IMU and image data
     * using a double-buffering strategy to avoid data drops.
     * 
     * Architecture:
     * - IMU Worker Thread: Continuously polls IMU measurements at high rate
     * - Image Worker Thread: Continuously grabs stereo images
     * - get_data() runs on caller thread, swaps buffers and converts to tensors
     * 
     * Double-buffering ensures that while data is being read/converted,
     * the worker threads can continue writing to the alternate buffer.
     */
    class ZedDataLoader {
    public:
        /// IMU buffer capacity (number of measurements)
        static constexpr std::size_t kIMUBufferCapacity = 512;

        /// Type alias for IMU circular buffer
        using IMUBuffer = CircularBuffer<IMU_Measurement, kIMUBufferCapacity>;

        /// Return type for IMU data: (accelerometer[N,3], gyroscope[N,3], timestamps[N])
        using IMUDataResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

        /// Return type for image data: (stereo_image[2,H,W,C], timestamp_ns)
        using ImageDataResult = std::tuple<torch::Tensor, uint64_t>;

        /// Return type for get_data(): (optional<IMU>, optional<Image>)
        using DataResult = std::tuple<
            std::optional<IMUDataResult>,
            std::optional<ImageDataResult>
        >;

    public:
        /**
         * @brief Construct a ZedDataLoader with specified settings.
         * @param resolution Camera resolution (default: HD720)
         * @param exposure Exposure time in microseconds. -1 for auto exposure (default).
         *                 Valid range depends on resolution/fps but typically 28-30000.
         * @param gain Sensor gain. -1 for auto gain (default). Valid range: 0-100.
         * @param fps Target frames per second. -1 for maximum FPS (default).
         *            If positive, the image worker will sleep to maintain this rate.
         */
        explicit ZedDataLoader(
            Resolution resolution = Resolution::HD720,
            int exposure = -1,
            int gain = -1,
            int fps = -1
        );

        /**
         * @brief Destructor. Stops worker threads and closes camera.
         */
        ~ZedDataLoader();

        // Non-copyable, non-movable
        ZedDataLoader(const ZedDataLoader&) = delete;
        ZedDataLoader& operator=(const ZedDataLoader&) = delete;
        ZedDataLoader(ZedDataLoader&&) = delete;
        ZedDataLoader& operator=(ZedDataLoader&&) = delete;

        /**
         * @brief Start the worker threads for IMU and image capture.
         * @throws CameraOpenError if camera fails to open
         */
        void start();

        /**
         * @brief Stop the worker threads and close the camera.
         */
        void stop();

        /**
         * @brief Check if the data loader is currently running.
         */
        bool is_running() const;

        /**
         * @brief Get the latest IMU and image data.
         * 
         * This method:
         * 1. Locks and swaps both IMU and image buffers
         * 2. Converts the buffered data to PyTorch tensors
         * 3. Clears the read buffers
         * 4. Unlocks the buffers
         * 
         * @return Tuple of (optional<IMUDataResult>, optional<ImageDataResult>)
         *         - IMUDataResult: (accel[N,3], gyro[N,3], timestamps[N]) or nullopt if no data
         *         - ImageDataResult: (stereo[2,H,W,C], timestamp_ns) or nullopt if no data
         */
        DataResult get_data();

        /**
         * @brief Get the left camera intrinsic matrix (3x3).
         * @return Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or nullopt if not started.
         */
        std::optional<torch::Tensor> left_intrinsics() const;

        /**
         * @brief Get the right camera intrinsic matrix (3x3).
         * @return Camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] or nullopt if not started.
         */
        std::optional<torch::Tensor> right_intrinsics() const;

        /**
         * @brief Get the left camera distortion coefficients.
         * @return Distortion coefficients [k1, k2, p1, p2, k3] or nullopt if not started.
         */
        std::optional<torch::Tensor> left_distortion() const;

        /**
         * @brief Get the right camera distortion coefficients.
         * @return Distortion coefficients [k1, k2, p1, p2, k3] or nullopt if not started.
         */
        std::optional<torch::Tensor> right_distortion() const;

        /**
         * @brief Get the stereo camera baseline in meters.
         * @return Baseline distance in meters or nullopt if not started.
         */
        std::optional<float> baseline() const;

        /**
         * @brief Get accelerometer sensor parameters.
         * @return Accelerometer parameters or nullopt if not started.
         */
        std::optional<SensorParameters> accelerometer_params() const;

        /**
         * @brief Get gyroscope sensor parameters.
         * @return Gyroscope parameters or nullopt if not started.
         */
        std::optional<SensorParameters> gyroscope_params() const;

        /**
         * @brief Get T_BS for the left camera (4x4): Body ← Sensor.
         * 
         * Returns the transformation matrix that transforms points FROM the
         * left camera frame TO the body frame (IMU).
         * 
         * Convention: T_BS means "Body ← Sensor" (sensor-to-body transform).
         * 
         * @return 4x4 homogeneous transformation matrix or nullopt if not started.
         */
        std::optional<torch::Tensor> left_T_BS() const;

        /**
         * @brief Get T_BS for the right camera (4x4): Body ← Sensor.
         * 
         * Returns the transformation matrix that transforms points FROM the
         * right camera frame TO the body frame (IMU).
         * 
         * Convention: T_BS means "Body ← Sensor" (sensor-to-body transform).
         * 
         * @return 4x4 homogeneous transformation matrix or nullopt if not started.
         */
        std::optional<torch::Tensor> right_T_BS() const;

        /**
         * @brief Get T_BS for the IMU (4x4): Body ← Sensor.
         * 
         * Since the IMU is the body frame, this is the identity matrix.
         * 
         * @return 4x4 identity matrix or nullopt if not started.
         */
        std::optional<torch::Tensor> imu_T_BS() const;

    private:
        /// Worker loop for IMU polling
        void imuWorkerLoop();

        /// Worker loop for image grabbing
        void imageWorkerLoop();

        /// Convert IMU buffer to tensors
        static std::optional<IMUDataResult> convertIMUToTensors(IMUBuffer& buffer);

        /// Convert stereo image buffer to tensor
        static std::optional<ImageDataResult> convertImageToTensor(StereoImageBuffer& buffer);

        /// Extract and store camera calibration parameters
        void extractCalibration();

    private:
        // Camera configuration
        sl::Camera camera_;
        Resolution resolution_;
        int exposure_;  ///< -1 for auto
        int gain_;      ///< -1 for auto
        int fps_;       ///< Target FPS, -1 for max
        std::chrono::nanoseconds frame_duration_{0};  ///< Target duration per frame (0 = no limit)
        sl::InitParameters init_params_;

        // Double buffers
        DoubleBuffer<IMUBuffer> imu_double_buffer_;
        DoubleBuffer<StereoImageBuffer> image_double_buffer_;

        // Worker threads
        std::thread imu_worker_thread_;
        std::thread image_worker_thread_;

        // Shutdown signal
        std::atomic<bool> shutdown_{false};
        std::atomic<bool> running_{false};

        // Camera calibration (populated in start())
        std::optional<torch::Tensor> left_intrinsics_;
        std::optional<torch::Tensor> right_intrinsics_;
        std::optional<torch::Tensor> left_distortion_;
        std::optional<torch::Tensor> right_distortion_;
        std::optional<float> baseline_;  ///< Stereo baseline in meters

        // IMU sensor parameters (populated in start())
        std::optional<SensorParameters> accelerometer_params_;
        std::optional<SensorParameters> gyroscope_params_;

        // Body-to-sensor transforms (populated in start())
        std::optional<torch::Tensor> left_T_BS_;   ///< IMU to left camera (4x4)
        std::optional<torch::Tensor> right_T_BS_;  ///< IMU to right camera (4x4)
        std::optional<torch::Tensor> imu_T_BS_;    ///< IMU to IMU = identity (4x4)
    };

} // namespace zed_loader
