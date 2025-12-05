#ifndef NN_HH
#define NN_HH

#include "layers.hh"
#include "kernels.hh"

class NN {
public:
    Linear fc1;
    Linear fc2;

    NN() : fc1(784,128), fc2(128,10) {}

    Tensor forward(const Tensor& x) {

        Tensor h1 = fc1.forward(x);  // First linear layer
        reLu(h1);                    // Activation

        Tensor out = fc2.forward(h1); // Second linear layer
        softmax(out);                 // Softmax

        return out;                   // Final prediction
    }
};

#endif
