#include "src/network.cu"
#include "src/mnistloader.hh"

int main(){
    auto train_images = loadMNISTImages(const std::string& path);
    auto train_labels = loadMNISTLabels("train-labels.idx1-ubyte");

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