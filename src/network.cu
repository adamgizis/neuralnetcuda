#ifndef NN_HH
#define NN_HH

#include "layers.hh"
#include "kernels.hh"
#include <vector>

class NN {
public:
    Linear fc1;
    Linear fc2;

    // Store activations for backward
    Tensor h1_raw;  // pre-ReLU
    Tensor h1;      // post-ReLU
    Tensor logits;

    NN() 
        : fc1(784,128), fc2(128,10),
          h1_raw({1,128}), h1({1,128}), logits({1,10}) {} 

    // Forward pass
    Tensor& forward(const Tensor& x) {
        Tensor out1 = fc1.forward(x);  // pre-ReLU

        // Resize h1_raw and h1 if batch size changed
        if (h1_raw.shape() != out1.shape()) {
            h1_raw = Tensor(out1.shape());
            h1 = Tensor(out1.shape());
        }
        h1_raw.copyFrom(out1);
        h1.copyFrom(out1);
        reLu(h1);  // in-place ReLU

        Tensor out2 = fc2.forward(h1);

        if (logits.shape() != out2.shape())
            logits = Tensor(out2.shape());
        logits.copyFrom(out2);

        return logits;
    }

    // Backward pass
    void backward(const Tensor& x, const std::vector<int>& labels) {
        int batch = x.shape()[0];

        Tensor grad_out({(size_t)batch, 10});
        softmaxCrossEntropyBackward(logits, labels, grad_out);

        Tensor grad_h1 = fc2.backward(h1, grad_out);
        Tensor grad_h1_relu = reLuBackward(h1_raw, grad_h1);
        fc1.backward(x, grad_h1_relu);
    }

    // update weights (calls for layer)
    void step(float lr) {
        fc1.step(lr);
        fc2.step(lr);
    }
};

#endif
