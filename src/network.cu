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
    Tensor backward(
        const Tensor& x,
        const Tensor& h1,
        const Tensor& logits,
        const std::vector<int>& labels,
        float lr)
    {
        int batch = (int)x.shape()[0];

        // 1. Compute grad_out for softmax+cross-entropy
        Tensor grad_out({(size_t)batch, 10});
        softmaxCrossEntropyBackward(logits, labels, grad_out); // pass raw logits

        // 2. Backprop through second layer
        Tensor grad_h1 = fc2.backward(h1, grad_out);

        // 3. ReLU backward
        Tensor grad_h1_relu = reLuBackward(h1, grad_h1);

        // 4. Backprop through first layer
        Tensor grad_x = fc1.backward(x, grad_h1_relu);

        // 5. Update weights
        fc1.step(lr);
        fc2.step(lr);

        return grad_x;
    }

};

#endif
