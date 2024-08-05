#ifndef MATRIX_H
#define MATRIX_H

#include <memory>
#include <functional>

struct Matrix {
    int height;
    int width;
    std::unique_ptr<float[]> data;
    Matrix(int height_, int width_, std::function<void(float*, int)> init_func) 
    : height(height_), width(width_), data(std::make_unique<float[]>(height_ * width_)) {
        init_func(data.get(), height_ * width_);
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(4);
        for(int i =0; i < height; ++i) {
            for(int j = 0; j < width; ++j) {
                std::cout << data[i * width + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif //MATRIX_H