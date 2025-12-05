#ifndef KERNELS_HH
#define KERNELS_HH

#include <cuda_runtime.h>
#include "tensor.hh"  

__global__ void addKernel(const float* a, const float* b, float* out, int size);

__global__ void matmulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K);

__global__ void addBiasKernel(float* out, const float* bias,
                              int batch, int out_dim);

__global__ void reLuKernel(float* x, int size);

__global__ void softmaxKernel(float* x, int rows, int cols);


void tensorAdd(const Tensor& A, const Tensor& B, Tensor& Out);

Tensor matmul(const Tensor& A, const Tensor& B);

void reLu(Tensor& t);

void softmax(Tensor& t);

#endif
