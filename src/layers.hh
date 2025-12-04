#pragma once

#include "tensor.hh"
#include "kernels.cu"

class Linear{
public:
    Tensor weight; // out, in 
    Tensor bias;

    Linear(size_t in, size_t out)
        : weight({out,in}), bias({out})
    {
        // initalize weights to be random to start
        for(size_t i = 0; i < weight.size(); i++){
            weight[i] = (rand()/ float(RAND_MAX) - 0.5f) * 0.1f;

        }

        // bias starts at 0
        for(size_t i = 0 ; i < bias.size(); i++){
            bias[i] = 0.0f;
        }

        weight.toDevice();
        bias.toDevice();

    }

    // forward 
    Tensor forward(const Tensor& x){
        // x: (batch, in)
        auto out = matmul(x,weight); // shape (batch, out)

        addBias(out);

        return out;
    }

private:
    void addBias(Tensor&out);


}


