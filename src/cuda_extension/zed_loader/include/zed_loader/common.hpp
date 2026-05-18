#pragma once

#include <string>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <iomanip>

// ATen / PyTorch
#include <torch/torch.h>
#include <ATen/ATen.h>

// ZED SDK
#include <sl/Camera.hpp>

namespace zed_loader {
    // Type aliases
    using Tensor = torch::Tensor;

    /* IMU Measurement at one time stamp */
    struct IMU_Measurement {
        /* linear acceleration [m/s^2] */
        sl::float3      linear_acceleration;

        /* angular velocity [deg/s] */
        sl::float3      angular_velocity;

        /* 3x3 covariance matrix for linear acceleration [m/s^2]^2 */
        sl::Matrix3f    linear_acceleration_cov;

        /* 3x3 covariance matrix for angular velocity [deg/s]^2 */
        sl::Matrix3f    angular_velocity_cov;

        /* Timestamp of measurement made in ns */
        uint64_t         timestamp_ns;

        IMU_Measurement& operator=(const sl::SensorsData& sensor_data) {
            linear_acceleration = sensor_data.imu.linear_acceleration;
            angular_velocity    = sensor_data.imu.angular_velocity;
            linear_acceleration_cov = sensor_data.imu.linear_acceleration_covariance;
            angular_velocity_cov    = sensor_data.imu.angular_velocity_covariance;
            timestamp_ns            = sensor_data.imu.timestamp.getNanoseconds();
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const IMU_Measurement& imu) {
            os << std::fixed << std::setprecision(6);
            os << "IMU_Measurement {\n";
            os << "  timestamp_ns: " << imu.timestamp_ns << "\n";
            os << "  linear_acceleration [m/s²]: ("
               << imu.linear_acceleration.x << ", "
               << imu.linear_acceleration.y << ", "
               << imu.linear_acceleration.z << ")\n";
            os << "  angular_velocity [deg/s]:   ("
               << imu.angular_velocity.x << ", "
               << imu.angular_velocity.y << ", "
               << imu.angular_velocity.z << ")\n";
            os << "  linear_acceleration_cov:\n";
            for (int i = 0; i < 3; ++i) {
                os << "    [" << imu.linear_acceleration_cov.r[i * 3 + 0] << ", "
                             << imu.linear_acceleration_cov.r[i * 3 + 1] << ", "
                             << imu.linear_acceleration_cov.r[i * 3 + 2] << "]\n";
            }
            os << "  angular_velocity_cov:\n";
            for (int i = 0; i < 3; ++i) {
                os << "    [" << imu.angular_velocity_cov.r[i * 3 + 0] << ", "
                             << imu.angular_velocity_cov.r[i * 3 + 1] << ", "
                             << imu.angular_velocity_cov.r[i * 3 + 2] << "]\n";
            }
            os << "}";
            return os;
        }
    };

    template <typename T, std::size_t Capacity>
    struct CircularBuffer {
        static_assert(Capacity > 0, "CircularBuffer capacity must be > 0.");
        static_assert(std::is_copy_constructible<T>::value && std::is_copy_assignable<T>::value,
                      "CircularBuffer requires copyable types.");

        void push(const T &data) {
            const std::size_t index = (head_ + size_) % Capacity;
            buffer_[index] = data;

            if (size_ < Capacity) size_++;
            else head_ = (head_ + 1) % Capacity;
        }

        std::size_t length() const {
            return size_;
        }

        void clear() {
            head_ = 0;
            size_ = 0;
        }

        T pop() {
            assert(size_ > 0);
            const T value = buffer_[head_];
            head_ = (head_ + 1) % Capacity;
            size_--;
            return value;
        }

    private:
        std::array<T, Capacity> buffer_{};
        std::size_t head_ = 0;
        std::size_t size_ = 0;
    };

    /**
     * @brief Double-buffer for lock-free producer/consumer pattern.
     * 
     * Workers write to the active buffer. When the consumer calls lock_and_swap(),
     * it atomically swaps the active buffer index and returns a reference to the
     * previously active buffer for reading. This avoids data drops when reading.
     * 
     * @tparam Buffer The buffer type (e.g., CircularBuffer<T, N>)
     */
    template <typename Buffer>
    class DoubleBuffer {
    public:
        DoubleBuffer() = default;
        
        // Non-copyable, non-movable (contains mutex)
        DoubleBuffer(const DoubleBuffer&) = delete;
        DoubleBuffer& operator=(const DoubleBuffer&) = delete;
        DoubleBuffer(DoubleBuffer&&) = delete;
        DoubleBuffer& operator=(DoubleBuffer&&) = delete;

        /**
         * @brief Get the current write buffer for the producer.
         * 
         * If the write buffer is currently locked by a consumer, this will
         * return the alternate buffer to avoid blocking.
         */
        Buffer& write_buffer() {
            int idx = write_index_.load(std::memory_order_acquire);
            // If current write buffer is locked, use the other one
            if (locked_index_.load(std::memory_order_acquire) == idx) {
                idx = 1 - idx;
            }
            return buffers_[idx];
        }

        /**
         * @brief Lock the current write buffer and swap to the other.
         * 
         * This atomically marks the current write buffer as locked,
         * swaps the write index to the other buffer, and returns a
         * reference to the locked buffer for reading.
         * 
         * @return Reference to the locked buffer (safe to read from)
         */
        Buffer& lock_and_swap() {
            std::lock_guard<std::mutex> lock(swap_mutex_);
            int current = write_index_.load(std::memory_order_acquire);
            locked_index_.store(current, std::memory_order_release);
            write_index_.store(1 - current, std::memory_order_release);
            return buffers_[current];
        }

        /**
         * @brief Unlock the previously locked buffer.
         */
        void unlock() {
            locked_index_.store(-1, std::memory_order_release);
        }

        /**
         * @brief Check if a buffer is currently locked.
         */
        bool is_locked() const {
            return locked_index_.load(std::memory_order_acquire) != -1;
        }

    private:
        std::array<Buffer, 2> buffers_{};
        std::atomic<int> write_index_{0};
        std::atomic<int> locked_index_{-1};  // -1 means no buffer is locked
        std::mutex swap_mutex_;
    };

    /**
     * @brief Timestamped image buffer for stereo images.
     * 
     * Stores a stereo image pair (left + right) along with the capture timestamp.
     */
    struct StereoImageBuffer {
        sl::Mat left_image;
        sl::Mat right_image;
        uint64_t timestamp_ns{0};
        bool has_data{false};

        void clear() {
            has_data = false;
            timestamp_ns = 0;
        }
    };

    // Base exception for all ZedLoader errors
    class ZedLoaderError : public std::runtime_error {
    public:
        explicit ZedLoaderError(const std::string& msg) 
            : std::runtime_error(msg) {}
    };
    
    // Specific exception types
    class CameraNotFoundError : public ZedLoaderError {
    public:
        explicit CameraNotFoundError(const std::string& msg = "ZED camera not found")
            : ZedLoaderError(msg) {}
    };
    
    class CameraOpenError : public ZedLoaderError {
    public:
        explicit CameraOpenError(const std::string& msg = "Failed to open ZED camera")
            : ZedLoaderError(msg) {}
    };
    
    class InvalidParameterError : public ZedLoaderError {
    public:
        explicit InvalidParameterError(const std::string& msg)
            : ZedLoaderError(msg) {}
    };

    class InternalError : public ZedLoaderError {
    public:
        explicit InternalError(const std::string& msg)
            : ZedLoaderError(msg) {}
    };

    bool reportCameraStatus(sl::Camera &camera);

    bool retrieve_latest_imu_reading(sl::Camera &zed_camera, IMU_Measurement &measurement);

} // namespace zed_loader
