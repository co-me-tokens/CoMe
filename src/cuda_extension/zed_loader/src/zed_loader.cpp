#include "zed_loader/zed_loader.hpp"

#include <chrono>

namespace zed_loader {

    // ============================================================================
    // Free functions
    // ============================================================================

    bool reportCameraStatus(sl::Camera &camera){
        if (!camera.isOpened()) {
            std::cerr << "Camera not opened! Cannot retrieve camera status." << std::endl;
            return false;
        } else {
            auto info = camera.getCameraInformation();
            std::cout << "Camera Model: " << info.camera_model << std::endl;
            std::cout << "Serial Number: " << info.serial_number << std::endl;
            // std::cout << "Camera Firmware: " << info.camera_configuration.firmware_version << std::endl;
            // std::cout << "Sensors Firmware: " << info.sensors_configuration.firmware_version << std::endl;

            // Display accelerometer sensor configuration
            sl::SensorParameters& accelerometer_params = info.sensors_configuration.accelerometer_parameters;
            std::cout << "Sensor Type: " << accelerometer_params.type << std::endl;
            std::cout << "      Sampling Rate: " << accelerometer_params.sampling_rate << std::endl;
            std::cout << "      Range: "       << accelerometer_params.range << std::endl;
            std::cout << "      Resolution: "  << accelerometer_params.resolution << std::endl;
            std::cout << "      Noise Density: " << accelerometer_params.noise_density << std::endl;
            std::cout << "      Random Walk: " << accelerometer_params.random_walk << std::endl;

            // Display gyroscope sensor configuration
            sl::SensorParameters& gyroscope_params = info.sensors_configuration.gyroscope_parameters;
            std::cout << "Sensor Type: " << gyroscope_params.type << std::endl;
            std::cout << "      Sampling Rate: " << gyroscope_params.sampling_rate << std::endl;
            std::cout << "      Range: "       << gyroscope_params.range << std::endl;
            std::cout << "      Resolution: "  << gyroscope_params.resolution << std::endl;
            std::cout << "      Noise Density: " << gyroscope_params.noise_density << std::endl;
            std::cout << "      Random Walk: " << gyroscope_params.random_walk << std::endl;
            return true;
        }
    }

    bool retrieve_latest_imu_reading(sl::Camera &camera, IMU_Measurement &measurement) {
        static sl::SensorsData zed_sensor_data_buffer;

        camera.getSensorsData(zed_sensor_data_buffer, sl::TIME_REFERENCE::CURRENT);
        const uint64_t prev_timestamp = measurement.timestamp_ns;
        const uint64_t curr_timestamp = zed_sensor_data_buffer.imu.timestamp.getNanoseconds();

        if (curr_timestamp > prev_timestamp) {
            measurement = zed_sensor_data_buffer;
            return true;
        }
        return false;
    }

    // ============================================================================
    // ZedDataLoader Implementation
    // ============================================================================

    ZedDataLoader::ZedDataLoader(Resolution resolution, int exposure, int gain, int fps)
        : resolution_(resolution), exposure_(exposure), gain_(gain), fps_(fps) {
        // Configure init parameters
        init_params_.camera_resolution = to_sl_resolution(resolution_);
        init_params_.depth_mode = sl::DEPTH_MODE::NONE;
        init_params_.enable_image_validity_check = false;
        
        // Calculate frame duration for FPS limiting
        if (fps_ > 0) {
            frame_duration_ = std::chrono::nanoseconds(1'000'000'000 / fps_);
        }
    }

    ZedDataLoader::~ZedDataLoader() {
        stop();
    }

    void ZedDataLoader::start() {
        if (running_.load(std::memory_order_acquire)) {
            return;  // Already running
        }

        std::cout << "[ZedDataLoader] Opening camera..." << std::endl;
        
        // Open camera
        auto err = camera_.open(init_params_);
        if (err != sl::ERROR_CODE::SUCCESS) {
            throw CameraOpenError("Failed to open ZED camera: " + 
                                  std::string(sl::toString(err)));
        }
        
        std::cout << "[ZedDataLoader] Camera opened successfully" << std::endl;

        // Apply exposure and gain settings
        if (exposure_ < 0 && gain_ < 0) {
            // Both auto - enable auto exposure/gain
            camera_.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 1);
            std::cout << "[ZedDataLoader] Using auto exposure/gain" << std::endl;
        } else {
            // Disable auto exposure/gain for manual control
            camera_.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0);
            
            if (exposure_ >= 0) {
                camera_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, exposure_);
                std::cout << "[ZedDataLoader] Exposure set to: " << exposure_ << " us" << std::endl;
            }
            if (gain_ >= 0) {
                camera_.setCameraSettings(sl::VIDEO_SETTINGS::GAIN, gain_);
                std::cout << "[ZedDataLoader] Gain set to: " << gain_ << std::endl;
            }
        }

        // Log FPS setting
        if (fps_ > 0) {
            std::cout << "[ZedDataLoader] FPS limit set to: " << fps_ << std::endl;
        } else {
            std::cout << "[ZedDataLoader] FPS limit: disabled (max rate)" << std::endl;
        }

        // Extract camera calibration parameters
        extractCalibration();

        // Report camera status
        reportCameraStatus(camera_);

        // Reset shutdown flag and set running
        shutdown_.store(false, std::memory_order_release);
        running_.store(true, std::memory_order_release);

        // Start worker threads
        imu_worker_thread_ = std::thread(&ZedDataLoader::imuWorkerLoop, this);
        image_worker_thread_ = std::thread(&ZedDataLoader::imageWorkerLoop, this);

        std::cout << "[ZedDataLoader] Started worker threads" << std::endl;
    }

    void ZedDataLoader::stop() {
        if (!running_.load(std::memory_order_acquire)) {
            return;  // Not running
        }

        // Signal shutdown
        shutdown_.store(true, std::memory_order_release);

        // Join worker threads
        if (imu_worker_thread_.joinable()) {
            imu_worker_thread_.join();
        }
        if (image_worker_thread_.joinable()) {
            image_worker_thread_.join();
        }

        // Close camera
        camera_.close();

        // Clear calibration data
        left_intrinsics_.reset();
        right_intrinsics_.reset();
        left_distortion_.reset();
        right_distortion_.reset();
        baseline_.reset();
        accelerometer_params_.reset();
        gyroscope_params_.reset();
        left_T_BS_.reset();
        right_T_BS_.reset();
        imu_T_BS_.reset();

        running_.store(false, std::memory_order_release);
        std::cout << "[ZedDataLoader] Stopped worker threads" << std::endl;
    }

    bool ZedDataLoader::is_running() const {
        return running_.load(std::memory_order_acquire);
    }

    void ZedDataLoader::imuWorkerLoop() {
        IMU_Measurement measurement{};
        measurement.timestamp_ns = 0;  // Initialize to 0 so first reading is always new

        while (!shutdown_.load(std::memory_order_relaxed)) {
            // Try to get new IMU data
            if (retrieve_latest_imu_reading(camera_, measurement)) {
                // Push to the current write buffer
                IMUBuffer& buffer = imu_double_buffer_.write_buffer();
                buffer.push(measurement);
            }
            // IMU polling is very fast, no need for explicit sleep
        }
    }

    void ZedDataLoader::imageWorkerLoop() {
        std::cout << "[ImageWorker] Starting image worker loop" << std::endl;
        if (fps_ > 0) {
            std::cout << "[ImageWorker] FPS limit: " << fps_ << std::endl;
        }
        
        sl::RuntimeParameters runtime_params;
        runtime_params.enable_fill_mode = false;

        while (!shutdown_.load(std::memory_order_relaxed)) {
            auto frame_start = std::chrono::steady_clock::now();
            
            // Grab a new frame
            auto grab_status = camera_.grab(runtime_params);
            if (grab_status == sl::ERROR_CODE::CORRUPTED_FRAME) {
                std::cerr << "[ImageWorker] grab() failed: " << sl::toString(grab_status) << std::endl;
                continue;
            }
            if (grab_status != sl::ERROR_CODE::SUCCESS) {
                std::cerr << "[ImageWorker] grab() failed: " << sl::toString(grab_status) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // Get the current write buffer
            StereoImageBuffer& buffer = image_double_buffer_.write_buffer();

            // Retrieve left and right rectified images
            camera_.retrieveImage(buffer.left_image, sl::VIEW::LEFT, sl::MEM::CPU);
            camera_.retrieveImage(buffer.right_image, sl::VIEW::RIGHT, sl::MEM::CPU);

            // Get timestamp
            buffer.timestamp_ns = camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
            buffer.has_data = true;
            
            // FPS limiting: sleep to maintain target frame rate
            if (frame_duration_.count() > 0) {
                auto frame_end = std::chrono::steady_clock::now();
                auto elapsed = frame_end - frame_start;
                if (elapsed < frame_duration_) {
                    std::this_thread::sleep_for(frame_duration_ - elapsed);
                }
            }
        }
    }

    ZedDataLoader::DataResult ZedDataLoader::get_data() {
        // Lock and swap both buffers
        IMUBuffer& imu_buffer = imu_double_buffer_.lock_and_swap();
        StereoImageBuffer& image_buffer = image_double_buffer_.lock_and_swap();

        // Convert to tensors
        auto imu_result = convertIMUToTensors(imu_buffer);
        auto image_result = convertImageToTensor(image_buffer);

        // Clear the buffers after reading
        imu_buffer.clear();
        image_buffer.clear();

        // Unlock buffers
        imu_double_buffer_.unlock();
        image_double_buffer_.unlock();

        return std::make_tuple(std::move(imu_result), std::move(image_result));
    }

    std::optional<ZedDataLoader::IMUDataResult> 
    ZedDataLoader::convertIMUToTensors(IMUBuffer& buffer) {
        const std::size_t count = buffer.length();
        if (count == 0) {
            return std::nullopt;
        }

        // Create tensors
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor accel = torch::zeros({static_cast<int64_t>(count), 3}, options);
        torch::Tensor gyro = torch::zeros({static_cast<int64_t>(count), 3}, options);
        torch::Tensor timestamps = torch::zeros({static_cast<int64_t>(count)}, 
                                                 torch::TensorOptions().dtype(torch::kInt64));

        // Access raw data pointers
        auto accel_accessor = accel.accessor<float, 2>();
        auto gyro_accessor = gyro.accessor<float, 2>();
        auto ts_accessor = timestamps.accessor<int64_t, 1>();

        // Pop all measurements and fill tensors
        std::size_t idx = 0;
        while (buffer.length() > 0) {
            IMU_Measurement m = buffer.pop();
            
            accel_accessor[idx][0] = m.linear_acceleration.x;
            accel_accessor[idx][1] = m.linear_acceleration.y;
            accel_accessor[idx][2] = m.linear_acceleration.z;

            gyro_accessor[idx][0] = m.angular_velocity.x;
            gyro_accessor[idx][1] = m.angular_velocity.y;
            gyro_accessor[idx][2] = m.angular_velocity.z;

            ts_accessor[idx] = static_cast<int64_t>(m.timestamp_ns);

            ++idx;
        }

        return std::make_tuple(std::move(accel), std::move(gyro), std::move(timestamps));
    }

    std::optional<ZedDataLoader::ImageDataResult>
    ZedDataLoader::convertImageToTensor(StereoImageBuffer& buffer) {
        if (!buffer.has_data) {
            return std::nullopt;
        }

        // Get image dimensions
        const int height = static_cast<int>(buffer.left_image.getHeight());
        const int width = static_cast<int>(buffer.left_image.getWidth());
        const int channels = static_cast<int>(buffer.left_image.getChannels());

        // Create tensor [2, H, W, C] for stereo images
        auto options = torch::TensorOptions().dtype(torch::kUInt8);
        torch::Tensor stereo = torch::zeros({2, height, width, channels}, options);

        // Copy left image data
        auto left_ptr = buffer.left_image.getPtr<sl::uchar1>();
        std::memcpy(stereo[0].data_ptr(), left_ptr, height * width * channels);

        // Copy right image data
        auto right_ptr = buffer.right_image.getPtr<sl::uchar1>();
        std::memcpy(stereo[1].data_ptr(), right_ptr, height * width * channels);

        uint64_t timestamp = buffer.timestamp_ns;

        return std::make_tuple(std::move(stereo), timestamp);
    }

    void ZedDataLoader::extractCalibration() {
        auto info = camera_.getCameraInformation();
        auto& calib = info.camera_configuration.calibration_parameters;
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Left camera intrinsics (3x3 matrix)
        {
            auto& cam = calib.left_cam;
            torch::Tensor K = torch::zeros({3, 3}, options);
            auto acc = K.accessor<float, 2>();
            acc[0][0] = cam.fx;  acc[0][1] = 0.0f;   acc[0][2] = cam.cx;
            acc[1][0] = 0.0f;    acc[1][1] = cam.fy; acc[1][2] = cam.cy;
            acc[2][0] = 0.0f;    acc[2][1] = 0.0f;   acc[2][2] = 1.0f;
            left_intrinsics_ = std::move(K);
            
            // Distortion coefficients [k1, k2, p1, p2, k3]
            torch::Tensor D = torch::zeros({5}, options);
            auto d_acc = D.accessor<float, 1>();
            d_acc[0] = cam.disto[0];  // k1
            d_acc[1] = cam.disto[1];  // k2
            d_acc[2] = cam.disto[2];  // p1
            d_acc[3] = cam.disto[3];  // p2
            d_acc[4] = cam.disto[4];  // k3
            left_distortion_ = std::move(D);
        }
        
        // Right camera intrinsics (3x3 matrix)
        {
            auto& cam = calib.right_cam;
            torch::Tensor K = torch::zeros({3, 3}, options);
            auto acc = K.accessor<float, 2>();
            acc[0][0] = cam.fx;  acc[0][1] = 0.0f;   acc[0][2] = cam.cx;
            acc[1][0] = 0.0f;    acc[1][1] = cam.fy; acc[1][2] = cam.cy;
            acc[2][0] = 0.0f;    acc[2][1] = 0.0f;   acc[2][2] = 1.0f;
            right_intrinsics_ = std::move(K);
            
            // Distortion coefficients [k1, k2, p1, p2, k3]
            torch::Tensor D = torch::zeros({5}, options);
            auto d_acc = D.accessor<float, 1>();
            d_acc[0] = cam.disto[0];  // k1
            d_acc[1] = cam.disto[1];  // k2
            d_acc[2] = cam.disto[2];  // p1
            d_acc[3] = cam.disto[3];  // p2
            d_acc[4] = cam.disto[4];  // k3
            right_distortion_ = std::move(D);
        }

        // Stereo baseline in meters
        baseline_ = calib.getCameraBaseline();

        // IMU sensor parameters
        accelerometer_params_ = SensorParameters::from_sl(
            info.sensors_configuration.accelerometer_parameters);
        gyroscope_params_ = SensorParameters::from_sl(
            info.sensors_configuration.gyroscope_parameters);
        
        // T_BS transforms: "Body ← Sensor" convention (transforms points FROM sensor TO body)
        // Body frame = IMU frame
        
        // IMU T_BS is identity (IMU is the body frame, so T_imu_imu = I)
        imu_T_BS_ = torch::eye(4, options);
        
        // Get camera-IMU transform from sensor configuration
        // ZED SDK provides T_leftcam_imu (transforms points from IMU to left camera)
        // We need T_imu_leftcam = inverse(T_leftcam_imu) for T_BS convention
        sl::Transform& cam_imu_transform = info.sensors_configuration.camera_imu_transform;
        {
            // First copy T_leftcam_imu from ZED SDK
            torch::Tensor T_cam_imu = torch::zeros({4, 4}, options);
            auto acc = T_cam_imu.accessor<float, 2>();
            // sl::Transform stores as column-major, we access row-by-row
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    acc[row][col] = cam_imu_transform.m[col * 4 + row];
                }
            }
            // Invert to get T_imu_leftcam (Body ← Sensor)
            left_T_BS_ = torch::inverse(T_cam_imu);
        }
        
        // Right camera T_BS = T_imu_rightcam
        // ZED SDK stereo_transform is T_rightcam_leftcam
        // T_rightcam_imu = T_rightcam_leftcam @ T_leftcam_imu
        // T_imu_rightcam = inverse(T_rightcam_imu)
        sl::Transform& stereo_transform = calib.stereo_transform;
        {
            // Build T_rightcam_leftcam from stereo transform
            torch::Tensor T_right_left = torch::zeros({4, 4}, options);
            auto stereo_acc = T_right_left.accessor<float, 2>();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    stereo_acc[row][col] = stereo_transform.m[col * 4 + row];
                }
            }
            
            // Build T_leftcam_imu again for computing T_rightcam_imu
            torch::Tensor T_left_imu = torch::zeros({4, 4}, options);
            auto left_acc = T_left_imu.accessor<float, 2>();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    left_acc[row][col] = cam_imu_transform.m[col * 4 + row];
                }
            }
            
            // T_rightcam_imu = T_rightcam_leftcam @ T_leftcam_imu
            torch::Tensor T_right_imu = torch::matmul(T_right_left, T_left_imu);
            
            // Invert to get T_imu_rightcam (Body ← Sensor)
            right_T_BS_ = torch::inverse(T_right_imu);
        }
        
        std::cout << "[ZedDataLoader] Extracted camera and IMU calibration parameters" << std::endl;
    }

    std::optional<torch::Tensor> ZedDataLoader::left_intrinsics() const {
        return left_intrinsics_;
    }

    std::optional<torch::Tensor> ZedDataLoader::right_intrinsics() const {
        return right_intrinsics_;
    }

    std::optional<torch::Tensor> ZedDataLoader::left_distortion() const {
        return left_distortion_;
    }

    std::optional<torch::Tensor> ZedDataLoader::right_distortion() const {
        return right_distortion_;
    }

    std::optional<float> ZedDataLoader::baseline() const {
        return baseline_;
    }

    std::optional<SensorParameters> ZedDataLoader::accelerometer_params() const {
        return accelerometer_params_;
    }

    std::optional<SensorParameters> ZedDataLoader::gyroscope_params() const {
        return gyroscope_params_;
    }

    std::optional<torch::Tensor> ZedDataLoader::left_T_BS() const {
        return left_T_BS_;
    }

    std::optional<torch::Tensor> ZedDataLoader::right_T_BS() const {
        return right_T_BS_;
    }

    std::optional<torch::Tensor> ZedDataLoader::imu_T_BS() const {
        return imu_T_BS_;
    }

} // namespace zed_loader
