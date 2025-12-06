#pragma once
#include <vector>
#include "tensor.hh"


#define WARP_SIZE 32
#define TM 4       // rows per thread in warp tile
#define TN 4       // cols per thread in warp tile
#define WMITER 2   // warptile iterations in M
#define WNITER 2   // warptile iterations in N
#define TILE_K 16  // K dimension tile size

#define TILE_WIDTH 16 // for tiled


// Elementwise add
void tensorAdd(const Tensor& A, const Tensor& B, Tensor& Out);

// Matrix multiply
Tensor matmul(const Tensor& A, const Tensor& B);

// ReLU activation
void reLu(Tensor& t);

// ReLU backward
Tensor reLuBackward(const Tensor& x, const Tensor& grad_out);

void softmax(Tensor& t);


void softmaxCrossEntropyBackward(
    const Tensor& softmax_out,
    const std::vector<int>& labels,
    Tensor& grad);

// Reduce sum across rows
void reduceSumRows(const Tensor& src, Tensor& dst);

// (y += alpha * x)
void axpy(const Tensor& x, Tensor& y, float alpha);

// called in the linear layer
__global__ void addBiasKernel(float* out, const float* bias, int batch, int out_dim);

// different multiplication techniques
__global__ void matmulWarpTiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);

__global__ void matmulKernelTiled(const float* A, const float* B, float* C, int M, int N, int K);