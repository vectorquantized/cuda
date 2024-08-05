#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "cuda_utils.h"


int main_saxpy() {
    int N = 1 << 20;
    float a = 3.0f;
    float *b, *c;

    CUDA_ERROR_CHECK(cudaMallocManaged(&b, N * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&c, N * sizeof(float)));

    for (int i = 0; i < N; ++i) {
        b[i] = 1.0f;
        c[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    saxpy<<<numBlocks, blockSize>>>(a, b, c, N);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    float maxError = 0.0f;
    for (int i=0; i < N; ++i) {
        maxError = std::max(maxError, c[i] - 5.0f);
    }
    std::cout << "Max Error: " << maxError << std::endl;

    CUDA_ERROR_CHECK(cudaFree(b));
    CUDA_ERROR_CHECK(cudaFree(c));

    return 0;
}