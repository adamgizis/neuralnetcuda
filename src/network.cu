#include "layers.hh"
#include "kernels.cu"

class NN {
public:
    Linear fc1;
    Linear fc2;

    NN(): fc1(784,128), fc2(128,10) {}

    Tensor forward(const Tensor){

        Tensor h1 = fc1.forward(x);
        reLu(h1);
        Tensor out = fc2.forward(h1);
        softmax(out);
        return out;
    }
};