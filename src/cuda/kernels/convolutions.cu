

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include "csrc/utils.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "cuda/cuda_utils.h"
#include "convolutions.h"

__constant__ float Conv_Mask_C[FILTER_SIZE * FILTER_SIZE];

__global__ void conv2d(float* matrix, float* conv_mask, float* output, int filter_radius, int width, int height) {

    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;

    int filter_size = 2 * filter_radius + 1;

    float p_value = 0.0f;
    if(row_out < height && col_out < width) {
        for(int i = 0; i < filter_size; ++i) {
            for(int j = 0; j < filter_size; ++j) {
                int row_in = row_out - filter_radius + i;
                int col_in = col_out - filter_radius + j;
                if(row_in >= 0 && row_in < height &&
                   col_in >= 0 && col_in < width ) {
                    p_value += conv_mask[i * filter_size + j] * matrix[row_in * width + col_in];
                }
            }
        }
        output[row_out * width + col_out] = p_value;
    }

}

__global__ void conv2d_tiled(const float* __restrict__ matrix, float* output, int width, int height) {

    int row_out = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
    int col_out = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;


    int row_in = row_out - FILTER_RADIUS;
    int col_in = col_out - FILTER_RADIUS;

    __shared__ float m_shared[IN_TILE_WIDTH][IN_TILE_WIDTH];

    // index into shared memory.
    int row_shared = threadIdx.y;
    int col_shared = threadIdx.x;

    // only copy valid regions from HBM.
    if (row_in >= 0 && row_in < height &&
        col_in >= 0  && col_in < width) {
            m_shared[row_shared][col_shared] = matrix[row_in * width + col_in];
        } else { // Halo elements
            m_shared[row_shared][col_shared] = 0.0f;
        }
    
    __syncthreads();

    // Convolve only if current thread index is within the bounds of output.
    if(row_shared < OUT_TILE_WIDTH && col_shared < OUT_TILE_WIDTH) {
        float p_value = 0.0f;
        for(int i = 0; i < FILTER_SIZE; ++i) {
            for(int j = 0; j < FILTER_SIZE; ++j) {
                // IN_TILE_WIDTH = OUT_TILE_WIDTH + FILTER_SIZE
                // m_shared is padded, the values of threadIdx and blockIdx that resulted in
                // row_in and col_in to fall out of bounds will have 0.0f stored in m_shared.
                p_value += Conv_Mask_C[i * FILTER_SIZE + j] * m_shared[row_shared + i][col_shared + j];
            }
        }
        if (row_out < height && col_out < width) {
            output[row_out * width + col_out] = p_value;
        }

    }

}

namespace conv2d_kernels {
void launch_kernel(float* matrix_d, float* output_d, int width, int height) {
    TIMED_CUDA_FUNCTION();
    // need IN_TILE_WIDTH threads to index into the shared memory.
    int block_size_x = IN_TILE_WIDTH;
    int block_size_y = IN_TILE_WIDTH;
    // num blocks in a grid will depend on the size of OUT_TILE_WIDTH
    int block_dim_x = (width + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH;
    int block_dim_y = (height + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH;
    dim3 threads_per_block(block_size_x, block_size_y, 1);
    dim3 blocks_per_grid(block_dim_x, block_dim_y, 1);

    conv2d_tiled<<<blocks_per_grid, threads_per_block>>>(matrix_d, output_d, width, height);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

}

void launch() {
    int M = 512;
    int N = 512;
    int k = FILTER_SIZE;
    int r = FILTER_RADIUS;

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
    // CUDA_ERROR_CHECK(cudaMalloc((void**) &conv_mask_d, mask_bytes));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, matrix_bytes));

    CUDA_ERROR_CHECK(cudaMemcpy(matrix_d, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
    // CUDA_ERROR_CHECK(cudaMemcpy(conv_mask_d, conv_mask.data(), mask_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(Conv_Mask_C, conv_mask.data(), mask_bytes));


    float hostConvMask[FILTER_SIZE * FILTER_SIZE];
    cudaMemcpyFromSymbol(hostConvMask, Conv_Mask_C, mask_bytes);

    bool is_mask_same = matrix::compare_matrices(hostConvMask, conv_mask.data(), k, k);
    std::cout << "is_mask_same: " << is_mask_same << std::endl;
    
    launch_kernel(matrix_d, output_d, M, N);

    CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, matrix_bytes, cudaMemcpyDeviceToHost));

    if (matrix::compare_matrices(output_h, cpu_output.data(), M, N)) {
        std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
    } else {
        std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
    }

    delete[] output_h;
    cudaFree(matrix_d);
    // cudaFree(conv_mask_d);
    cudaFree(output_d);
}
}

