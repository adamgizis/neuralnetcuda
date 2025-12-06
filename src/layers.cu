// src/layers.cu
#include "layers.hh"
#include "kernels.hh"



 // Backward pass
Tensor Linear::backward(const Tensor& x, const Tensor& grad_out) {
    Tensor x_T = x.transpose();
    Tensor gw = matmul(x_T, grad_out);
    grad_w.copyFrom(gw);  // <-- must use copyFrom

    reduceSumRows(grad_out, grad_b); 
    // grad_b.copyFrom(gb);  <-- if you compute gb as a Tensor

    Tensor W_T = weights.transpose();
    Tensor grad_x = matmul(grad_out, W_T);

    return grad_x;
}

void Linear::step(float lr) {
    axpy(grad_w, weights, -lr);
    axpy(grad_b, bias, -lr);

    // reset gradients
    for (size_t i = 0; i < grad_w.size(); i++) grad_w[i] = 0.f;
    for (size_t i = 0; i < grad_b.size(); i++) grad_b[i] = 0.f;

    weights.toDevice();
    bias.toDevice();
}
Tensor Linear::forward(const Tensor& x) {
    Tensor out = matmul(x, weights);
    addBias(out);
    return out;
}

void Linear::addBias(Tensor& out) {
    int batch = (int)out.shape()[0];
    int out_dim = (int)out.shape()[1];
    int size = batch * out_dim;

    int block = 256;
    int grid = (size + block - 1) / block;

    addBiasKernel<<<grid, block>>>(out.device(), bias.device(), batch, out_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
