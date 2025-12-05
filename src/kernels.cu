 #include <cuda_runtime.h>
#include <cmath>
#include "kernels.hh"
#include "layers.hh"  

__global__
void addKernel(const float* a, const float* b, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

void tensorAdd(const Tensor& A, const Tensor& B, Tensor& Out)
{
    int size  = A.size();
    int block = 256;
    int grid  = (size + block - 1) / block;

    addKernel<<<grid, block>>>(A.device(), B.device(), Out.device(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


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

    int M = A.shape()[0];
    int K = A.shape()[1];
    int B_rows = B.shape()[0];
    int N = B.shape()[1];

    if (K != B_rows) {
        std::cerr << "ERROR: matmul dimension mismatch: "
                << "A: " << M << "x" << K
                << ", B: " << B_rows << "x" << N << std::endl;
        exit(1);
    }

    Tensor C({(size_t)M, (size_t)N});

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmulKernel<<<grid, block>>>(A.device(), B.device(), C.device(), M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return C;
}

__global__
void addBiasKernel(float* out, const float* bias, int batch, int out_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch * out_dim) {
        int col = i % out_dim;
        out[i] += bias[col];
    }
}

void Linear::addBias(Tensor& out)
{
    int batch   = out.shape()[0];
    int out_dim = out.shape()[1];
    int size    = batch * out_dim;

    int block = 256;
    int grid  = (size + block - 1) / block;

    addBiasKernel<<<grid, block>>>(out.device(), bias.device(), batch, out_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


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
    int size  = t.size();
    int block = 256;
    int grid  = (size + block - 1) / block;

    reLuKernel<<<grid, block>>>(t.device(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__
void softmaxKernel(float* x, int rows, int cols)
{
    int row = blockIdx.x;

    if (row >= rows)
        return;

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
    int rows = t.shape()[0];
    int cols = t.shape()[1];

    softmaxKernel<<<rows, 1>>>(t.device(), rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
