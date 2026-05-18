#include "zed_loader/zed_loader.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <vector>

#include <rerun.hpp>


namespace {
    std::atomic_bool g_shutdown{false};

    void HandleSigInt(int) {
        g_shutdown.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Log IMU data tensors to Rerun
     */
    void logIMUToRerun(
        rerun::RecordingStream& rec,
        const torch::Tensor& accel,
        const torch::Tensor& gyro,
        const torch::Tensor& timestamps
    ) {
        const int64_t count = accel.size(0);
        if (count == 0) return;

        // Convert tensors to vectors for rerun
        auto accel_accessor = accel.accessor<float, 2>();
        auto gyro_accessor = gyro.accessor<float, 2>();
        auto ts_accessor = timestamps.accessor<int64_t, 1>();

        std::vector<int64_t> ts_vec(count);
        std::vector<float> accel_x(count), accel_y(count), accel_z(count);
        std::vector<float> gyro_x(count), gyro_y(count), gyro_z(count);

        for (int64_t i = 0; i < count; ++i) {
            ts_vec[i] = ts_accessor[i];
            accel_x[i] = accel_accessor[i][0];
            accel_y[i] = accel_accessor[i][1];
            accel_z[i] = accel_accessor[i][2];
            gyro_x[i] = gyro_accessor[i][0];
            gyro_y[i] = gyro_accessor[i][1];
            gyro_z[i] = gyro_accessor[i][2];
        }

        // Log accelerometer data
        rec.send_columns(
            "imu/accelerometer/x",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(accel_x).columns()
        );
        rec.send_columns(
            "imu/accelerometer/y",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(accel_y).columns()
        );
        rec.send_columns(
            "imu/accelerometer/z",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(accel_z).columns()
        );

        // Log gyroscope data
        rec.send_columns(
            "imu/gyroscope/x",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(gyro_x).columns()
        );
        rec.send_columns(
            "imu/gyroscope/y",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(gyro_y).columns()
        );
        rec.send_columns(
            "imu/gyroscope/z",
            rerun::TimeColumn::from_nanos_since_epoch("timestamp", ts_vec),
            rerun::Scalars(gyro_z).columns()
        );
    }

    /**
     * @brief Log stereo image tensor to Rerun
     * 
     * The stereo tensor is expected to have shape [2, H, W, C] where:
     * - Index 0 is the left image
     * - Index 1 is the right image
     * - C is typically 4 (BGRA from ZED SDK)
     */
    void logStereoImageToRerun(
        rerun::RecordingStream& rec,
        const torch::Tensor& stereo_image,
        uint64_t timestamp_ns
    ) {
        // stereo_image shape: [2, H, W, C]
        const uint32_t height = static_cast<uint32_t>(stereo_image.size(1));
        const uint32_t width = static_cast<uint32_t>(stereo_image.size(2));
        const int64_t channels = stereo_image.size(3);

        // Set the timestamp for this frame
        rec.set_time_timestamp_nanos_since_epoch("timestamp", static_cast<int64_t>(timestamp_ns));

        // Extract left and right images
        // The tensor is contiguous in memory, so we can get pointers to each image
        auto left_image = stereo_image[0].contiguous();
        auto right_image = stereo_image[1].contiguous();

        const uint8_t* left_ptr = left_image.data_ptr<uint8_t>();
        const uint8_t* right_ptr = right_image.data_ptr<uint8_t>();

        const size_t image_size = static_cast<size_t>(height) * width * channels;

        // Determine the color model based on channel count
        // ZED SDK provides BGRA format (4 channels)
        if (channels == 4) {
            // Log as RGBA (rerun expects RGBA, so we need to swap B and R)
            // Create temporary buffers for RGBA conversion
            std::vector<uint8_t> left_rgba(image_size);
            std::vector<uint8_t> right_rgba(image_size);

            // Convert BGRA to RGBA
            for (size_t i = 0; i < image_size; i += 4) {
                left_rgba[i + 0] = left_ptr[i + 2];  // R <- B
                left_rgba[i + 1] = left_ptr[i + 1];  // G <- G
                left_rgba[i + 2] = left_ptr[i + 0];  // B <- R
                left_rgba[i + 3] = left_ptr[i + 3];  // A <- A

                right_rgba[i + 0] = right_ptr[i + 2];
                right_rgba[i + 1] = right_ptr[i + 1];
                right_rgba[i + 2] = right_ptr[i + 0];
                right_rgba[i + 3] = right_ptr[i + 3];
            }

            rec.log(
                "camera/left",
                rerun::Image::from_rgba32(left_rgba, {width, height})
            );
            rec.log(
                "camera/right",
                rerun::Image::from_rgba32(right_rgba, {width, height})
            );
        } else if (channels == 3) {
            // Assume BGR, convert to RGB
            std::vector<uint8_t> left_rgb(image_size);
            std::vector<uint8_t> right_rgb(image_size);

            for (size_t i = 0; i < image_size; i += 3) {
                left_rgb[i + 0] = left_ptr[i + 2];
                left_rgb[i + 1] = left_ptr[i + 1];
                left_rgb[i + 2] = left_ptr[i + 0];

                right_rgb[i + 0] = right_ptr[i + 2];
                right_rgb[i + 1] = right_ptr[i + 1];
                right_rgb[i + 2] = right_ptr[i + 0];
            }

            rec.log(
                "camera/left",
                rerun::Image::from_rgb24(left_rgb, {width, height})
            );
            rec.log(
                "camera/right",
                rerun::Image::from_rgb24(right_rgb, {width, height})
            );
        } else {
            // Grayscale or unexpected format - log raw bytes
            std::vector<uint8_t> left_data(left_ptr, left_ptr + image_size);
            std::vector<uint8_t> right_data(right_ptr, right_ptr + image_size);

            rec.log(
                "camera/left",
                rerun::Image::from_grayscale8(left_data, {width, height})
            );
            rec.log(
                "camera/right",
                rerun::Image::from_grayscale8(right_data, {width, height})
            );
        }
    }

} // namespace


int main() {
    std::signal(SIGINT, HandleSigInt);

    // Initialize Rerun recording stream and connect via gRPC
    rerun::RecordingStream rec("zed_data_loader_demo");
    auto rerun_err = rec.connect_grpc();
    if (rerun_err.is_err()) {
        std::cerr << "Failed to connect to Rerun viewer via gRPC" << std::endl;
        return 1;
    }
    std::cout << "[Rerun] Connected to viewer via gRPC" << std::endl;

    // Create and start the ZedDataLoader
    zed_loader::ZedDataLoader loader(zed_loader::Resolution::VGA);
    loader.start();

    // FPS tracking
    auto fps_start_time = std::chrono::steady_clock::now();
    std::size_t imu_sample_count = 0;
    std::size_t image_count = 0;
    constexpr double fps_update_interval_sec = 1.0;

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        // Get data from the loader (non-blocking, returns nullopt if no data)
        auto [imu_data, image_data] = loader.get_data();

        // Process IMU data if available
        if (imu_data.has_value()) {
            auto& [accel, gyro, timestamps] = *imu_data;
            imu_sample_count += accel.size(0);
            
            // Log to Rerun
            logIMUToRerun(rec, accel, gyro, timestamps);
        }

        // Process image data if available
        if (image_data.has_value()) {
            auto& [stereo_image, timestamp_ns] = *image_data;
            ++image_count;
            
            // Log stereo images to Rerun
            logStereoImageToRerun(rec, stereo_image, timestamp_ns);
        }

        // Calculate and display FPS
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - fps_start_time).count();
        if (elapsed_sec >= fps_update_interval_sec) {
            double imu_fps = static_cast<double>(imu_sample_count) / elapsed_sec;
            double img_fps = static_cast<double>(image_count) / elapsed_sec;
            std::cout << "[Stats] IMU: " << std::fixed << std::setprecision(1) << imu_fps 
                      << " samples/s, Images: " << img_fps << " fps" << std::endl;
            imu_sample_count = 0;
            image_count = 0;
            fps_start_time = now;
        }

        // Small sleep to avoid busy-waiting when no data is available
        if (!imu_data.has_value() && !image_data.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Stop the loader
    loader.stop();

    std::cout << "[Main] Shutdown complete" << std::endl;
    return 0;
}
