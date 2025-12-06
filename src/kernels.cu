// src/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "kernels.hh"
#include "layers.hh"

// ------------------------- utility macro already expected -------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err)
                  << " (" << expr << ") at " << file << ":" << line << std::endl;
        std::exit(1);
    }
}
#endif

// -----------------------------------------------------------------------------
// Elementwise add
// -----------------------------------------------------------------------------
__global__
void addKernel(const float* a, const float* b, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] + b[i];
}

void tensorAdd(const Tensor& A, const Tensor& B, Tensor& Out)
{
    int size  = (int)A.size();
    int block = 256;
    int grid  = (size + block - 1) / block;

    addKernel<<<grid, block>>>(A.device(), B.device(), Out.device(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// MatMul (naive) : C = A @ B
// A: M x K, B: K x N, C: M x N
// -----------------------------------------------------------------------------
__global__
void matmulKernel(const float* A, const float* B, float* C,
                  int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

Tensor matmul(const Tensor& A, const Tensor& B)
{
    int M = (int)A.shape()[0];
    int K = (int)A.shape()[1];
    int B_rows = (int)B.shape()[0];
    int N = (int)B.shape()[1];

    if (K != B_rows) {
        std::cerr << "ERROR: matmul dimension mismatch: "
                  << "A: " << M << "x" << K
                  << ", B: " << B_rows << "x" << N << std::endl;
        std::exit(1);
    }

    Tensor C({(size_t)M, (size_t)N});

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmulKernel<<<grid, block>>>(A.device(), B.device(), C.device(), M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return C;
}

// -----------------------------------------------------------------------------
// Add bias: out += bias (broadcast bias over batch rows)
// -----------------------------------------------------------------------------
__global__ void addBiasKernel(float* out, const float* bias, int batch, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * out_dim) {
        int col = i % out_dim;
        out[i] += bias[col];
    }
}
// -----------------------------------------------------------------------------
// ReLU forward (in-place)
// -----------------------------------------------------------------------------
__global__
void reLuKernel(float* x, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = fmaxf(0.f, x[i]);
    }
}

void reLu(Tensor& t)
{
    int size  = (int)t.size();
    int block = 256;
    int grid  = (size + block - 1) / block;

    reLuKernel<<<grid, block>>>(t.device(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// softmax forward (in-place, one block per row)
// -----------------------------------------------------------------------------
__global__
void softmaxKernel(float* x, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr = x + row * cols;

    // Find max (for numerical stability)
    float maxval = row_ptr[0];
    for (int i = 1; i < cols; i++) {
        maxval = fmaxf(maxval, row_ptr[i]);
    }

    // Exponentiate
    float sum = 0.f;
    for (int i = 0; i < cols; i++) {
        row_ptr[i] = expf(row_ptr[i] - maxval);
        sum += row_ptr[i];
    }

    // Normalize
    for (int i = 0; i < cols; i++) {
        row_ptr[i] /= sum;
    }
}

void softmax(Tensor& t)
{
    int rows = (int)t.shape()[0];
    int cols = (int)t.shape()[1];

    softmaxKernel<<<rows, 1>>>(t.device(), rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// ------------------ BACKPROP HELPERS ------------------
// -----------------------------------------------------------------------------

// ----------------- reLu backward -----------------
__global__
void reLuBackwardKernel(const float* x, const float* grad_out, float* grad_in, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_in[i] = (x[i] > 0.f) ? grad_out[i] : 0.f;
    }
}

Tensor reLuBackward(const Tensor& x, const Tensor& grad_out)
{
    Tensor grad_in(x.shape());
    int size = (int)x.size();
    int block = 256;
    int grid  = (size + block - 1) / block;

    reLuBackwardKernel<<<grid, block>>>(x.device(), grad_out.device(), grad_in.device(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_in;
}

// ----------------- reduceSumRows: sum over batch (rows) -> output vector length = cols -----------------
__global__
void reduceSumRowsKernel(const float* src, float* dst, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float s = 0.f;
    for (int r = 0; r < rows; ++r) {
        s += src[r * cols + col];
    }
    dst[col] = s;
}

void reduceSumRows(const Tensor& src, Tensor& dst)
{
    int rows = (int)src.shape()[0];
    int cols = (int)src.shape()[1];

    // expect dst shape == {cols}
    if (dst.size() != (size_t)cols) {
        std::cerr << "reduceSumRows: destination has wrong size\n";
        std::exit(1);
    }

    int block = 256;
    int grid  = (cols + block - 1) / block;

    reduceSumRowsKernel<<<grid, block>>>(src.device(), dst.device(), rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------- axpy: y += alpha * x -----------------
__global__
void axpyKernel(const float* x, float* y, float alpha, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) y[i] += alpha * x[i];
}

void axpy(const Tensor& x, Tensor& y, float alpha)
{
    int size = (int)x.size();
    if (y.size() != x.size()) {
        std::cerr << "axpy: size mismatch\n";
        std::exit(1);
    }

    int block = 256;
    int grid  = (size + block - 1) / block;

    axpyKernel<<<grid, block>>>(x.device(), y.device(), alpha, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// softmax + cross-entropy backward wrapper
// grad_logits must be preallocated with shape (rows, cols)
// labels is host-side vector<int> with length rows
// -----------------------------------------------------------------------------
__global__
void softmaxCrossEntropyBackwardKernel(
    const float* softmax_out, const int* labels,
    float* grad_logits, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int label = labels[row];
    int base = row * cols;

    for (int j = 0; j < cols; ++j) {
        int idx = base + j;
        float y = (j == label) ? 1.f : 0.f;
        grad_logits[idx] = softmax_out[idx] - y;
    }
}

void softmaxCrossEntropyBackward(const Tensor& softmax_out,
                                 const std::vector<int>& labels,
                                 Tensor& grad)
{
    int rows = (int)softmax_out.shape()[0];
    int cols = (int)softmax_out.shape()[1];

    if ((int)grad.shape()[0] != rows || (int)grad.shape()[1] != cols) {
        std::cerr << "softmaxCrossEntropyBackward: grad has wrong shape\n";
        std::exit(1);
    }

    // copy labels to device
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), rows * sizeof(int), cudaMemcpyHostToDevice));

    // launch: one block per row, kernel loops over columns
    softmaxCrossEntropyBackwardKernel<<<rows, 1>>>(
        softmax_out.device(), d_labels, grad.device(), rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_labels);
}