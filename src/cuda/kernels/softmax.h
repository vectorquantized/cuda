#ifndef SOFTMAX_H
#define SOFTMAX_H


#include <cuda_runtime.h>

__global__ void softmax(float* matrix, int M, int N);
void softmax_kernel_launch(float* matrix, int M, int N);

namespace softmax_kernels {
void launch();
};


#endif // SOFTMAX_H