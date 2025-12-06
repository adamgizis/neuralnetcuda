#pragma once
#include "tensor.hh"
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>


// must first run the python script 
inline uint32_t readBigEndian32(std::ifstream& f) {
    uint32_t x = 0;
    f.read(reinterpret_cast<char*>(&x), 4);
    return __builtin_bswap32(x);  // convert from big-endian
}

Tensor loadMNISTImages(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = readBigEndian32(f);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file!");

    uint32_t num_images = readBigEndian32(f);
    uint32_t num_rows   = readBigEndian32(f);
    uint32_t num_cols   = readBigEndian32(f);

    size_t image_size = num_rows * num_cols;

    // Output tensor: (num_images, 784)
    Tensor images({num_images, image_size});

    std::vector<uint8_t> buffer(image_size);

    for (size_t i = 0; i < num_images; i++) {
        f.read(reinterpret_cast<char*>(buffer.data()), image_size);

        for (size_t j = 0; j < image_size; j++) {
            images[i * image_size + j] = buffer[j] / 255.0f;  // normalize to [0,1]
        }
    }

    return images;
}

Tensor loadMNISTLabels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = readBigEndian32(f);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file!");

    uint32_t num_labels = readBigEndian32(f);

    Tensor labels({num_labels});   // 1D tensor storing label IDs
    std::vector<uint8_t> buffer(num_labels);

    f.read(reinterpret_cast<char*>(buffer.data()), num_labels);

    for (size_t i = 0; i < num_labels; i++)
        labels[i] = static_cast<float>(buffer[i]);

    return labels;
}
