#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <iostream>

//
// CUDA error-checking macro
//
#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err)
                  << " (" << expr << ") at " << file << ":" << line << std::endl;
        std::exit(1);
    }
}

class Tensor {
public:
    // Create tensor from {dim1, dim2, ...}
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape)
    {
        size_ = 1;
        for (size_t s : shape_) size_ *= s;

        // Allocate host memory
        host_ = new float[size_];

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&device_, size_ * sizeof(float)));
    }

    // Disable copying (deep copy can be added if needed)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move support
    Tensor(Tensor&& other) noexcept {
        moveFrom(other);
    }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            moveFrom(other);
        }
        return *this;
    }

    // Destructor
    ~Tensor() {
        cleanup();
    }

    //
    // Data access
    //
    float& operator[](size_t i) { return host_[i]; }
    const float& operator[](size_t i) const { return host_[i]; }

    float* host()   { return host_; }
    float* device() { return device_; }

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }

    //
    // Synchronization
    //
    void toDevice() {
        CUDA_CHECK(cudaMemcpy(device_, host_, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void toHost() {
        CUDA_CHECK(cudaMemcpy(host_, device_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    std::vector<size_t> shape_;
    size_t size_ = 0;

    float* host_   = nullptr;
    float* device_ = nullptr;

    void cleanup() {
        if (host_)   delete[] host_;
        if (device_) cudaFree(device_);
        host_ = nullptr;
        device_ = nullptr;
    }

    void moveFrom(Tensor& other) {
        shape_  = std::move(other.shape_);
        size_   = other.size_;
        host_   = other.host_;
        device_ = other.device_;

        other.host_   = nullptr;
        other.device_ = nullptr;
        other.size_   = 0;
    }
};

