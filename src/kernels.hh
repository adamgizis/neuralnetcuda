#pragma once
#include <vector>
#include "tensor.hh"

// ------------------------- Forward Declarations -------------------------

// Elementwise add
void tensorAdd(const Tensor& A, const Tensor& B, Tensor& Out);

// Matrix multiply
Tensor matmul(const Tensor& A, const Tensor& B);

// ReLU activation (in-place)
void reLu(Tensor& t);

// ReLU backward
Tensor reLuBackward(const Tensor& x, const Tensor& grad_out);

// Softmax (in-place)
void softmax(Tensor& t);

// Softmax + cross-entropy backward
void softmaxCrossEntropyBackward(
    const Tensor& softmax_out,
    const std::vector<int>& labels,
    Tensor& grad);

// Reduce sum across rows (sum rows into a single bias gradient)
void reduceSumRows(const Tensor& src, Tensor& dst);

// AXPY update (y += alpha * x)
void axpy(const Tensor& x, Tensor& y, float alpha);

// ------------------------- Linear helper (defined in kernels.cu) -------------------------
__global__ void addBiasKernel(float* out, const float* bias, int batch, int out_dim);