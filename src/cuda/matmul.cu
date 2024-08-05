#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cublas_v2.h>
#include "kernels.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "csrc/utils.h"

int main() {
    
    int M = 1024;
    int K = 1024;
    int N = 1024;
    Matrix mat1_h(M, K, random_init);
    Matrix mat2_h(K, N, random_init);
    float* out_h = new float[M * N];
    float* out_cpu = new float[M * N];
    matmul_cpu(mat1_h.data.get(), mat2_h.data.get(), out_cpu, M, K, N);
    float *mat1_d, *mat2_d, *out_d;

    CUDA_ERROR_CHECK(cudaMalloc((void**) &mat1_d, M * K * sizeof(float)));
    cudaMalloc((void**) &mat2_d, K * N * sizeof(float));
    cudaMalloc((void**) &out_d, M * N * sizeof(float));

    cudaMemcpy(mat1_d, mat1_h.data.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, mat2_h.data.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size_x = 16;
    int block_size_y = 16;

    dim3 threads_per_block(block_size_x, 
                           block_size_y);

    dim3 blocks_per_grid((N + block_size_x - 1) / block_size_x, 
                         (M + block_size_y - 1) / block_size_y);


    float alpha = 1.0f;
    float beta = 0.0f;
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record the start event
    cudaEventRecord(start);
    matmul_cuda<<<blocks_per_grid, threads_per_block>>>(mat1_d, mat2_d, out_d, M, K, N);
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, mat2_d, N, mat1_d, K, &beta, out_d, N);
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, mat2_d, K, mat1_d, M, &beta, out_d, N);
    
    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

     // Calculate the elapsed time between the start and stop events
     float milliseconds = 0;
     cudaEventElapsedTime(&milliseconds, start, stop);
 
     std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}