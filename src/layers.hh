#pragma once

#include "tensor.hh"
#include "kernels.hh"

class Linear {
public:
    Tensor weight; // in × out
    Tensor bias;

    Linear(size_t in, size_t out)
        : weight({in, out}), bias({out})
    {
        // initialize weights randomly
        for (size_t i = 0; i < weight.size(); i++) {
            weight[i] = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
        }

        // initialize bias to 0
        for (size_t i = 0; i < bias.size(); i++) {
            bias[i] = 0.0f;
        }

        weight.toDevice();
        bias.toDevice();
    }


    Tensor forward(const Tensor& x) {
        // x: (batch, in)
        // weight: (in, out)
        // output: (batch, out)
        Tensor out = matmul(x, weight);
        addBias(out);
        return out;
    }

private:
    void addBias(Tensor& out);
};
