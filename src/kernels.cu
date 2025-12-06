// src/kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "kernels.hh"
#include "layers.hh"


// Elementwise add
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


// MatMul (naive) : C = A @ B
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

// tiled
__global__ void matmulKernelTiled(const float* A, const float* B, float* C,
                                  int M, int N, int K)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // load into shared memory
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // multiply
        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
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



    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    // dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,  // number of blocks in x
    //             (M + TILE_WIDTH - 1) / TILE_WIDTH); // number of blocks in y

    // matmulKernelTiled<<<dimGrid, dimBlock>>>(A.device(), B.device(), C.device(), M, N, K);
    

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmulKernel<<<grid, block>>>(A.device(), B.device(), C.device(), M, N, K);



    // int threadsPerBlock = 128;  // 4 warps per block
    // dim3 blockDim(threadsPerBlock, 1, 1);
    // dim3 gridDim((N + WNITER*TN - 1) / (WNITER*TN), 
    //             (M + WMITER*TM - 1) / (WMITER*TM), 1);

    // size_t sharedMemBytes = TILE_K * (blockDim.y + blockDim.x) * sizeof(float);

    // matmulWarpTiled<<<gridDim, blockDim, sharedMemBytes>>>(A.device(), B.device(), C.device(), M, N, K);


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return C;
}

// asdd bias: out += bias (broadcast bias over batch rows)
__global__ void addBiasKernel(float* out, const float* bias, int batch, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * out_dim) {
        int col = i % out_dim;
        out[i] += bias[col];
    }
}

// ReLU forward (in-place)
// max of 0 and weight
__global__
void reLuKernel(float* x, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = fmaxf(0.f, x[i]);
    }
}



__global__ void matmulWarpTiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K)
{
    extern __shared__ float sharedMem[];
    float* As = sharedMem;                    // TILE_K x blockDim.y
    float* Bs = As + TILE_K * blockDim.y;    // TILE_K x blockDim.x

    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // Each warp computes a warptile starting at (warpRow, warpCol)
    int warpsPerRow = blockDim.x / WARP_SIZE;  // number of warps per block row
    int warpRow = warpId / warpsPerRow;
    int warpCol = warpId % warpsPerRow;

    int globalRow = blockIdx.y * blockDim.y + warpRow * (WMITER * TM);
    int globalCol = blockIdx.x * blockDim.x + warpCol * (WNITER * TN);

    // Registers to store thread subtiles
    float regM[WMITER * TM];
    float regN[WNITER * TN];
    float threadResults[WMITER * TM * WNITER * TN] = {0};

    // Loop over K tiles
    for (int tileK = 0; tileK < K; tileK += TILE_K) {
        // Load A and B tiles into shared memory
        int aRow = globalRow + laneId / TILE_K;
        int aCol = tileK + laneId % TILE_K;
        if (aRow < M && aCol < K)
            As[laneId] = A[aRow * K + aCol];
        else
            As[laneId] = 0.0f;

        int bRow = tileK + laneId / TILE_K;
        int bCol = globalCol + laneId % TILE_K;
        if (bRow < K && bCol < N)
            Bs[laneId] = B[bRow * N + bCol];
        else
            Bs[laneId] = 0.0f;

        __syncthreads();

        // Load thread's subtiles from shared memory into registers
        for (uint dotIdx = 0; dotIdx < TILE_K; ++dotIdx) {
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (uint i = 0; i < TM; ++i)
                    regM[wSubRowIdx * TM + i] =
                        As[(dotIdx * TM) + wSubRowIdx * TM + i];

            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for (uint i = 0; i < TN; ++i)
                    regN[wSubColIdx * TN + i] =
                        Bs[(dotIdx * TN) + wSubColIdx * TN + i];

            // Multiply-accumulate
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                          (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
        }

        __syncthreads();
    }

    // Write thread results to global memory
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int r = globalRow + wSubRowIdx * TM + resIdxM;
                    int c = globalCol + wSubColIdx * TN + resIdxN;
                    if (r < M && c < N)
                        C[r * N + c] = threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                                     (wSubColIdx * TN) + resIdxN];
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

// softmax forward (in-place, one block per row)
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


// backwards relu
__global__
void reLuBackwardKernel(const float* x, const float* grad_out, float* grad_in, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // either grad_out or 0 depending on value of x
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

// sum over batch (rows) -> output vector length = cols
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

// axpy: y += alpha * x 
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

// softmax + cross-entropy backward wrapper
// grad_logits must be preallocated with shape (rows, cols)
// labels is host-side vector<int> with length rows
__global__
void softmaxCrossEntropyBackwardKernel(const float* logits, const int* labels,
                                       float* grad_out, int batch, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    // offset for this sample
    const float* logit_row = logits + idx * num_classes;
    float* grad_row = grad_out + idx * num_classes;

    // Find max logit (for numerical stability)
    float max_logit = logit_row[0];
    for (int j = 1; j < num_classes; j++)
        if (logit_row[j] > max_logit) max_logit = logit_row[j];


    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        grad_row[j] = expf(logit_row[j] - max_logit);
        sum_exp += grad_row[j];
    }

    // normalize
    for (int j = 0; j < num_classes; j++)
        grad_row[j] /= sum_exp;

    // gradient of cross-entropy
    int label = labels[idx];
    grad_row[label] -= 1.0f;
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

    // one block per row, kernel loops over columns
    softmaxCrossEntropyBackwardKernel<<<rows, 1>>>(
        softmax_out.device(), d_labels, grad.device(), rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_labels);
}