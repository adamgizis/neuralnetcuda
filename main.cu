#include "src/network.cu"
#include "src/mnistloader.hh"

int main(){
    auto train_images = loadMNISTImages("mnist-dataset/train-images.idx3-ubyte");
    auto train_labels = loadMNISTLabels("mnist-dataset/train-labels.idx1-ubyte");

    NN net;
    // forward sample
    train_images.toDevice();
    Tensor logits = net.forward(train_images);
    logits.toHost();


    // print first sample
    for (int i = 0; i < 10; i++) {
        std::cout << logits[i] << " ";
    }
    std::cout << "\n";

}