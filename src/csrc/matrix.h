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

    // copy constructor, deep copies data value.
    // reason for this is that we support in-place operations on matrix
    // having this just makes it easier to validate the cpu vs gpu implementations.
    Matrix(const Matrix& other)
    : height(other.height), width(other.width),
    data(std::make_unique<float[]>(other.height * other.width)) {
        std::copy(other.data.get(), other.data.get() + other.height * other.width, data.get());
    }

    // TODO: add move semantics

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