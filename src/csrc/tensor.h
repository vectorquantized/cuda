#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>

template<typename T>
class Tensor {

    Tensor(int dim1, dim2, dim3)
    : dims{dim1, dim2, dim3}, is_4d(false) {
        data.resize(dim1 * dim2 * dim3);
    }

    Tensor(int dim1, dim2, dim3)
    : dims{dim1, dim2, dim3, dim4}, is_4d(true) {
        data.resize(dim1 * dim2 * dim3 * dim4);
    }

    T& operator() (int i, int j, int k) {
        if (is_4d) {
            throw std::runtime_error("4D tensor, use 4D operator.");
        }
        return data[i * dims[1] * dims[2] + j * dims[2] + k];
    }

    T& operator() (int i, int j, int k, int l) {
        if (!is_4d) {
            throw std::runtime_error("This is not a 4D tensor, use 3D operator.");
        }
        return data[i * dims[1] * dims[2] * dims[3] + j * dims[3] * dims[2] + k * dims[3] + l];
    }

    T& operator() (int i, int j, int k) const {
        if (is_4d) {
            throw std::runtime_error("4D tensor, use 4D operator.");
        }
        return data[i * dims[1] * dims[2] + j * dims[2] + k];
    }

    T& operator() (int i, int j, int k, int l) const {
        if (!is_4d) {
            throw std::runtime_error("This is not a 4D tensor, use 3D operator.");
        }
        return data[i * dims[1] * dims[2] * dims[3] + j * dims[3] * dims[2] + k * dims[3] + l];
    }

    std::vector<int> ndims() const {
        return dims;
    }

    int size() const {
        return data.size();
    }

private;
    std::vector<int> dims;
    std::vector<T> data;
    bool is_4d;

};

#endif // TENSOR_H