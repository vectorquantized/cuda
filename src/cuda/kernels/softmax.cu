#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include "csrc/utils.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "cuda/cuda_utils.h"
#include "softmax.h"

__global__ void softmax(float* matrix, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M) {
        float max_val = 1e-20;
        float sum = 0.0f;

        for(int col = 0; col < N; ++col) {
            max_val = max(max_val, matrix[row * N + col]);
        }

        for(int col = 0; col < N; ++col) {
            matrix[row * N + col] = expf(matrix[row * N + col] - max_val);
            sum += matrix[row* N + col]; 
        }

        for(int col = 0; col < N; ++col) {
            matrix[row * N + col] /= sum;
        }
    }
}


void softmax_kernel_launch(float* matrix, int M, int N) {
    TIMED_CUDA_FUNCTION();
    dim3 threads_per_block(1, 256);
    dim3 blocks_per_grid(1, M);
    softmax<<<blocks_per_grid, threads_per_block>>>(matrix, M, N);
    cudaDeviceSynchronize();
}

namespace softmax_kernels {
void launch() {

    int M = 8192;
    int N = 4096;
    int size_bytes = sizeof(float) * M * N;
    Matrix mat_h(M, N, random_init);
    Matrix mat_cpu(mat_h);
    softmax_cpu(mat_cpu.data.get(), M, N);
    float *mat_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**) & mat_d, size_bytes));
    CUDA_ERROR_CHECK(cudaMemcpy(mat_d, mat_h.data.get(), size_bytes, cudaMemcpyHostToDevice));

    softmax_kernel_launch(mat_d, M, N);

    float* out_h = new float[M * N];

    CUDA_ERROR_CHECK(cudaMemcpy(out_h, mat_d, size_bytes, cudaMemcpyDeviceToHost));

    if (matrix::compare_matrices(out_h, mat_cpu.data.get(), M, N)) {
        std::cout << "CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "CUDA kernel's result does NOT match the CPU result." << std::endl;
    }


    cudaFree(mat_d);
    delete[] out_h;
}
}
