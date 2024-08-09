#ifndef _KERNELS_IMPL
#define _KERNELS_IMPL

#include "kernels.h"

__global__ void saxpy_grid_strided(float a, float* b, float* c, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=index; i< N; i += stride) {
        c[i] += a * b[i];
    }
}

__global__ void saxpy(float a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] += a * b[i];
    }
}

__global__ void conv1d(float* matrix, float* conv_mask, float* output, int mask_width, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float p_value = 0.0f;
    int start = idx - mask_width / 2;
    for (int j = 0; j < mask_width; ++j) {
        if (start + j >= 0 && start + j < width) {
            p_value += matrix[start + j] * conv_mask[j];
        }
    }
    output[idx] = p_value;
}

__global__ void conv2d(float* matrix, float* conv_mask, float* output, int r, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float p_value = 0.0f;
    if (row < height && col < width) {
        for(int i = 0; i< 2 * r + 1; ++i) {
            for (int j = 0; j < 2 * r + 1; ++j) {
                int in_row = row - r + i;
                int in_col = col - r + j; 
                if (in_row >= 0 && in_row < height &&
                    in_col >= 0 && in_col < width) {
                        p_value += matrix[in_row * width + in_col] * conv_mask[i * (2 * r + 1) + j];
                    }
            }
        }
        output[row * width + col] = p_value;
    }
}

__global__ void rgbToGrayScale(float *in, float *out, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height) {
        int gray_offset = row * width + col;
        int rgb_offset = gray_offset * CHANNELS;
        float r = in[rgb_offset];
        float g = in[rgb_offset + 1];
        float b = in[rgb_offset + 2];
        out[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

__global__ void matmul_cuda(float* a, float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = value;
    }
}

__global__ void matmul_cuda_tiled(float* A, float* B, float* C, int M, int K, int N) {

    __shared__ float a_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_shared[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float p_value = 0.0f;
    
    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH ; ++i) {
        if (row < M && i * TILE_WIDTH + tx < K) {
            a_shared[ty][tx] = A[row * K + i * TILE_WIDTH + tx];
        } else {
            a_shared[ty][tx] = 0.0f;
        }
        if (col < N && i * TILE_WIDTH + ty < K) {
            b_shared[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col];
        } else {
            b_shared[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++ j) {
            p_value += a_shared[ty][j] * b_shared[j][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = p_value;
    }

}


#endif // _KERNELS_IMPL
