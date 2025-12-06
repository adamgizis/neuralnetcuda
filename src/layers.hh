#pragma once

#include "tensor.hh"
#include "kernels.hh"
#include <cmath>
#include <cstdlib>

class Linear {
public:
    Tensor weights;   // (in, out)
    Tensor bias;      // (out)
    Tensor grad_w;    // same shape as weights
    Tensor grad_b;    // same shape as bias

Linear(size_t in, size_t out)
    : weights({in, out}),
      bias({out}),
      grad_w({in, out}),
      grad_b({out})
{
    // He initialization for ReLU
    float limit = sqrtf(6.0f / (in + out)); 
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = (rand() / float(RAND_MAX) * 2 - 1) * limit;
    }

    // initialize bias to 0
    for (size_t i = 0; i < bias.size(); i++)
        bias[i] = 0.f;

    weights.toDevice();
    bias.toDevice();
}


    // Forward pass: x @ W + b
    Tensor forward(const Tensor& x);

    // Backward pass
    Tensor backward(const Tensor& x, const Tensor& grad_out);
    void step(float lr);

private:
    void addBias(Tensor& out);
};
