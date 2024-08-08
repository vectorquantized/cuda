
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cublas_v2.h>
#include "kernels.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "csrc/utils.h"

namespace entry {

void conv1d_kernel(float* matrix_d, float* conv_mask_d, float* output_d, int mask_width, int width) {
    TIMED_CUDA_FUNCTION();
    int block_size = 256;
    dim3 threads_per_block(block_size, 1, 1);
    dim3 blocks_per_grid((width + (block_size - 1))/block_size, 1, 1);

    conv1d<<<blocks_per_grid, threads_per_block>>>(matrix_d, conv_mask_d, output_d, mask_width, width);
    cudaDeviceSynchronize();

}

void matmul_kernel(float* mat1_d, float* mat2_d, float* out_d, int M, int K, int N) {
    TIMED_CUDA_FUNCTION();
    int block_size_x = 16;
    int block_size_y = 16;

    dim3 threads_per_block(block_size_x, 
                           block_size_y);

    dim3 blocks_per_grid((N + block_size_x - 1) / block_size_x, 
                         (M + block_size_y - 1) / block_size_y);

    matmul_cuda_tiled<<<blocks_per_grid, threads_per_block>>>(mat1_d, mat2_d, out_d, M, K, N);
    cudaDeviceSynchronize();
}

void conv2d_kernel(float* matrix_d, float* conv_mask_d, float* output_d, 
                  int r, int width, int height) {
    TIMED_CUDA_FUNCTION();
    int block_size_x = 16;
    int block_size_y = 16;
    int block_dim_x = width + (block_size_x - 1)/block_size_x;
    int block_dim_y = height + (block_size_y - 1)/block_size_y;
    dim3 threads_per_block(block_size_x, block_size_y, 1);
    dim3 blocks_per_grid(block_dim_x, block_dim_y, 1);

    conv2d<<<blocks_per_grid, threads_per_block>>>(matrix_d, conv_mask_d, output_d, r, width, height);
    cudaDeviceSynchronize();

}

void conv2d_kernel_invocation() {
    int M = 512;
    int N = 512;
    int k = 5;
    int r = (k - 1) / 2;

    std::vector<float> matrix(M * N);
    random_init(matrix.data(), matrix.size());
    std::vector<float> conv_mask(k * k);
    random_init(conv_mask.data(), conv_mask.size());
    std::vector<float> cpu_output(M * N, 0.0f);
    
    conv2d_cpu(matrix, conv_mask, cpu_output, r, M, N);

    float *matrix_d, *conv_mask_d, *output_d;
    int matrix_bytes = sizeof(float) * M * N;
    int mask_bytes = sizeof(float) * k * k;
    float *output_h = new float[M * N];

    CUDA_ERROR_CHECK(cudaMalloc((void**) &matrix_d, matrix_bytes));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &conv_mask_d, mask_bytes));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, matrix_bytes));

    CUDA_ERROR_CHECK(cudaMemcpy(matrix_d, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(conv_mask_d, conv_mask.data(), mask_bytes, cudaMemcpyHostToDevice));


    conv2d_kernel(matrix_d, conv_mask_d, output_d, r, M, N);

    CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, matrix_bytes, cudaMemcpyDeviceToHost));

    if (matrix::compare_matrices(output_h, cpu_output.data(), M, N)) {
        std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
    }

    delete[] output_h;
    cudaFree(matrix_d);
    cudaFree(conv_mask_d);
    cudaFree(output_d);
}

void conv1d_kernel_invocation() {
    int N = 1 << 20;
    std::vector<float> matrix_h(N);
    random_init(matrix_h.data(), N);
    std::vector<float> conv_mask_h = {0.2, 0.5, 0.2};
    int mask_width = conv_mask_h.size();
    int width = matrix_h.size();
    std::vector<float> cpu_output(width);

    conv1d_cpu(matrix_h, conv_mask_h, cpu_output, mask_width, width);

    float *matrix_d, *conv_mask_d, *output_d;

    float* output_h = new float[width];

    CUDA_ERROR_CHECK(cudaMalloc((void**) &matrix_d, sizeof(float) * width));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &conv_mask_d, sizeof(float) * mask_width));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, sizeof(float) * width));

    CUDA_ERROR_CHECK(cudaMemcpy(matrix_d, matrix_h.data(), sizeof(float) * width, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(conv_mask_d, conv_mask_h.data(), sizeof(float) * mask_width, cudaMemcpyHostToDevice));

    conv1d_kernel(matrix_d, conv_mask_d, output_d, mask_width, width);
    
    CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, sizeof(float) * width, cudaMemcpyDeviceToHost));

    if (utils::compare_vectors(output_h, cpu_output.data(), N)) {
        std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
    }
    
    delete[] output_h;
    cudaFree(matrix_d);
    cudaFree(conv_mask_d);
    cudaFree(output_d);
}


void matmul_invocation() {

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

    matmul_kernel(mat1_d, mat2_d, out_d, M, K, N);
    
    CUDA_ERROR_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (matrix::compare_matrices(out_h, out_cpu, M, N)) {
        std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
    }

    delete [] out_h;
    delete [] out_cpu;
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(out_d);
}
}
