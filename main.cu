#include <iostream>
#include <vector>
#include <cmath>
#include "src/network.cu"
#include "src/mnistloader.hh"

int main() {

    // load in (function normalizes for you)
    Tensor train_images = loadMNISTImages("mnist-dataset/train-images.idx3-ubyte");


    Tensor train_labels_tensor = loadMNISTLabels("mnist-dataset/train-labels.idx1-ubyte");

    std::vector<int> train_labels(train_labels_tensor.size());
    for (size_t i = 0; i < train_labels_tensor.size(); i++)
        train_labels[i] = static_cast<int>(train_labels_tensor[i]);

    const int num_samples = (int)train_labels.size();
    const int batch_size = 64;
    const float lr = 0.001f; // this seems to work
    const int epochs = 5;

    NN net;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << "\n";

        int correct = 0;
        float total_loss = 0.0f;

        for (int start = 0; start < num_samples; start += batch_size) {
            int end = std::min(start + batch_size, num_samples);
            int curr_batch = end - start;

            // Prepare input batch
            Tensor x({(size_t)curr_batch, 784});
            std::vector<int> labels(curr_batch);
            for (int i = 0; i < curr_batch; i++) {
                labels[i] = train_labels[start + i];
                for (int j = 0; j < 784; j++)
                    x[i * 784 + j] = train_images[(start + i) * 784 + j];
            }
            x.toDevice();

            // forward pass
            net.forward(x);  // fills net.h1 and net.logits

            // copy back to verify
            net.logits.toHost();

            // loss & accuracy
            float batch_loss = 0.f;
            for (int i = 0; i < curr_batch; i++) {
                int label = labels[i];

                // get max logit
                float max_logit = net.logits[i * 10];
                for (int j = 1; j < 10; j++)
                    max_logit = std::max(max_logit, net.logits[i * 10 + j]);

                // log-sum-exp
                float sum_exp = 0.f;
                for (int j = 0; j < 10; j++)
                    sum_exp += std::exp(net.logits[i * 10 + j] - max_logit);

                // Cross-entropy loss
                float log_prob = net.logits[i * 10 + label] - max_logit - std::log(sum_exp);
                batch_loss += -log_prob;

                // Accuracy
                int pred = 0;
                float max_val = net.logits[i * 10];
                for (int j = 1; j < 10; j++) {
                    if (net.logits[i * 10 + j] > max_val) {
                        max_val = net.logits[i * 10 + j];
                        pred = j;
                    }
                }
                if (pred == label) correct++;
            }
            total_loss += batch_loss / curr_batch;  // normalize per batch

            // backwards pass
            net.backward(x, labels);

            //step
            net.step(lr);
        }

        float accuracy = 100.f * correct / num_samples;
        float avg_loss = total_loss / (num_samples / batch_size); // average over batches

        std::cout << "Epoch " << epoch + 1
                << " | Loss: " << avg_loss
                << " | Accuracy: " << accuracy << "%\n";
    }

    std::cout << "Training finished.\n";
    return 0;
}
