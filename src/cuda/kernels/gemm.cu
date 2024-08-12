
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include "csrc/utils.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "cuda/utils/cuda_utils.h"
#include "gemm.h"

namespace gemm_kernels {


__global__ void gemm_cuda(const float* __restrict__ a, const float* __restrict__ b, 
                          float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float p_value = 0.0f;

        for(int i = 0; i < K; ++i) {
            p_value += a[row * K + i] * b[i * N + col];
        }

        c[row * M + col] = p_value;
    }
}

__global__ void gemm_cuda_tiled(const float* __restrict__  a, const float* __restrict__ b, 
                                float* c, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float a_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_shared[TILE_WIDTH][TILE_WIDTH];

    float p_value = 0.0f;
    for(int ph = 0; ph < (K + TILE_WIDTH - 1)/ TILE_WIDTH; ++ph) {
        if (row < M && ph * TILE_WIDTH + tx < K) {
            a_shared[ty][tx] = a[row * K + ph * TILE_WIDTH + tx];
        } else {
            a_shared[ty][tx] = 0.0f;
        }
        if(ph * TILE_WIDTH + ty < K && col < N) {
            b_shared[ty][tx] = b[(ph * TILE_WIDTH + ty) * N + col];
        } else {
            b_shared[ty][tx] = 0.0f;
        }
    
        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; ++i) {
            p_value += a_shared[ty][i] * b_shared[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N) {
        c[row * N + col] = p_value;
    }
}


void gemm_kernel_launch(float* mat1_d, float* mat2_d, float* out_d, int M, int K, int N) {
    TIMED_CUDA_FUNCTION();
    int block_size_x = TILE_WIDTH;
    int block_size_y = TILE_WIDTH;

    dim3 threads_per_block(block_size_x, 
                           block_size_y);

    dim3 blocks_per_grid((N + block_size_x - 1) / block_size_x, 
                         (M + block_size_y - 1) / block_size_y);

    gemm_cuda_tiled<<<blocks_per_grid, threads_per_block>>>(mat1_d, mat2_d, out_d, M, K, N);
    cudaDeviceSynchronize();
}

void launch() {

    int M = 2048;
    int K = 1024;
    int N = 2048;
    Matrix mat1_h(M, K, random_init);
    Matrix mat2_h(K, N, random_init);
    float* out_h = new float[M * N];
    float* out_cpu = new float[M * N];
    matmul_cpu(mat1_h.data.get(), mat2_h.data.get(), out_cpu, M, K, N);
    float *mat1_d, *mat2_d, *out_d;

    CUDA_ERROR_CHECK(cudaMalloc((void**) &mat1_d, M * K * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &mat2_d, K * N * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &out_d, M * N * sizeof(float)));

    CUDA_ERROR_CHECK(cudaMemcpy(mat1_d, mat1_h.data.get(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(mat2_d, mat2_h.data.get(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    gemm_kernel_launch(mat1_d, mat2_d, out_d, M, K, N);
    
    CUDA_ERROR_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (matrix::compare_matrices(out_h, out_cpu, M, N)) {
        std::cout << "CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "CUDA kernel's result does NOT match the CPU result." << std::endl;
    }

    delete [] out_h;
    delete [] out_cpu;
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(out_d);
}

}
