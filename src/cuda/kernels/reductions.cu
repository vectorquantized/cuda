#include <cuda_runtime.h>
#include "reductions.h"
#include "csrc/init_utils.h"
#include "cuda/utils/cuda_utils.h"
#include <numeric>
#include <iostream>


__global__ void add(float* input, float* output, int size) {

    extern __shared__ float s_data[];
    unsigned int t_idx = threadIdx.x;
    unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (g_idx < size) {
        s_data[t_idx] = input[g_idx];
    } else {
        s_data[t_idx] = 0.0f;
    }
    __syncthreads(); // barrier sync, we wait for all the data to be loaded in the SMEM.

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t_idx < stride) {
            s_data[t_idx] += s_data[t_idx + stride];
        }
        __syncthreads();
    }
    if (t_idx == 0) {
        output[blockIdx.x] = s_data[t_idx];
    }
}

bool close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}


namespace reduction_kernels {

void add_kernel_launch(float* input, float* output, int size, int threads_per_block, int blocks_per_grid) {
    TIMED_CUDA_FUNCTION();
    int shared_memory_size = threads_per_block * sizeof(float);
    add<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(input, output, size);
    cudaDeviceSynchronize();
}

void launch(std::string name) {
    int M = 8192;
    int threads_per_block = 256;
    int blocks_per_grid = (M + (threads_per_block - 1)) / threads_per_block;
    float* output_h = new float[blocks_per_grid]();
    std::vector<float> input_vec(M);

    random_init(input_vec.data(), M);

    float *input_d, *output_d;

    CUDA_ERROR_CHECK(cudaMalloc((void**) &input_d, sizeof(float) * M));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, sizeof(float) * blocks_per_grid));

    CUDA_ERROR_CHECK(cudaMemcpy(input_d, input_vec.data(), sizeof(float) * M, cudaMemcpyHostToDevice));

    add_kernel_launch(input_d, output_d, M, threads_per_block, blocks_per_grid);
    
    CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, sizeof(float) * blocks_per_grid, cudaMemcpyDeviceToHost));

    std::cout.precision(10);
    float cpu_sum = std::accumulate(input_vec.begin(), input_vec.end(), 0.0f);
    std::cout << "cpu: " << cpu_sum << std::endl;
    float gpu_sum = std::accumulate(output_h, output_h + blocks_per_grid, 0.0f);
    std::cout << "gpu: " << gpu_sum << std::endl;
    std::cout << "diff: " << fabs(cpu_sum - gpu_sum) << std::endl;
    if (close(cpu_sum, gpu_sum)) {
        std::cout << "CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cout << "CUDA kernel's result does NOT match the CPU result." << std::endl;
    }

    CUDA_ERROR_CHECK(cudaFree(input_d));
    CUDA_ERROR_CHECK(cudaFree(output_d));
    delete output_h;


}
}
