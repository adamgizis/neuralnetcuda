#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "src/network.cu"
#include "src/mnistloader.hh"

int main() {
    // Load MNIST dataset
    Tensor train_images = loadMNISTImages("mnist-dataset/train-images.idx3-ubyte");
    Tensor train_labels_tensor = loadMNISTLabels("mnist-dataset/train-labels.idx1-ubyte");

    // Convert labels Tensor -> std::vector<int>
    std::vector<int> train_labels(train_labels_tensor.size());
    for (size_t i = 0; i < train_labels_tensor.size(); i++)
        train_labels[i] = static_cast<int>(train_labels_tensor[i]);

    const int num_samples = static_cast<int>(train_labels.size());
    const int batch_size = 64;
    const float lr = 0.1f;
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

            // Forward pass
            Tensor h1 = net.fc1.forward(x);
            reLu(h1);

            // Sanity check: print first batch activations after ReLU
            if (start == 0 && epoch == 0) {
                h1.toHost();
                std::cout << "Sample h1 after ReLU:\n";
                for (int i = 0; i < 10; i++) std::cout << h1[i] << " ";
                std::cout << "\n";
            }

            Tensor logits = net.fc2.forward(h1); // raw logits
            logits.toHost();                       // copy to host for loss/accuracy

            // Sanity check: print sample logits
            if (start == 0 && epoch == 0) {
                std::cout << "Sample logits (first 10) of first batch:\n";
                for (int i = 0; i < 10; i++) std::cout << logits[i] << " ";
                std::cout << "\n";
            }

            // Compute numerically stable cross-entropy loss + accuracy
            for (int i = 0; i < curr_batch; i++) {
                int label = labels[i];

                // 1. Find max logit for numerical stability
                float max_logit = logits[i * 10];
                for (int j = 1; j < 10; j++)
                    max_logit = std::max(max_logit, logits[i * 10 + j]);

                // 2. Compute log-sum-exp
                float sum_exp = 0.f;
                for (int j = 0; j < 10; j++)
                    sum_exp += std::exp(logits[i * 10 + j] - max_logit);

                // 3. Loss contribution
                float log_prob = logits[i * 10 + label] - max_logit - std::log(sum_exp);
                total_loss += -log_prob;

                // 4. Accuracy
                int pred = 0;
                float max_val = logits[i * 10];
                for (int j = 1; j < 10; j++) {
                    if (logits[i * 10 + j] > max_val) {
                        max_val = logits[i * 10 + j];
                        pred = j;
                    }
                }
                if (pred == label) correct++;
            }

            // Backward pass
            Tensor grad_x = net.backward(x, h1, logits, labels, lr);

            // Sanity check: print sum of gradients for first batch
            if (start == 0 && epoch == 0) {
                net.fc1.grad_w.toHost();
                net.fc1.grad_b.toHost();
                float sum_grad_w = 0.f, sum_grad_b = 0.f;
                for (size_t i = 0; i < net.fc1.grad_w.size(); i++) sum_grad_w += net.fc1.grad_w[i];
                for (size_t i = 0; i < net.fc1.grad_b.size(); i++) sum_grad_b += net.fc1.grad_b[i];
                std::cout << "Sum grad_w (fc1) first batch: " << sum_grad_w << "\n";
                std::cout << "Sum grad_b (fc1) first batch: " << sum_grad_b << "\n";
            }
        }

        float accuracy = static_cast<float>(correct) / num_samples * 100.0f;
        float avg_loss = total_loss / num_samples;

        std::cout << "Epoch " << epoch + 1
                  << " | Loss: " << avg_loss
                  << " | Accuracy: " << accuracy << "%\n";
    }

    std::cout << "Training finished.\n";
    return 0;
}
