#ifndef KERNELS_H
#define KERNELS_H

#define CHANNELS 3
#define TILE_WIDTH 16
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * (FILTER_RADIUS) + 1)
#define OUT_TILE_WIDTH 12
#define IN_TILE_WIDTH ((OUT_TILE_WIDTH) + (FILTER_SIZE) - 1)

#include "cuda_utils.h"

__global__ void matmul_cuda(float* a, float* b, float* c, int M, int K, int N);
__global__ void matmul_cuda_tiled(float* A, float* B, float* C, int M, int K, int N);
__global__ void rgbToGrayScale(float *in, float *out, int width, int height);
__global__ void saxpy_grid_strided(float a, float* b, float* c, int N);
__global__ void saxpy(float a, float* b, float* c, int N);
__global__ void conv1d(float* matrix, float* conv_mask, float* output, int mask_width, int width);
__global__ void conv2d(float* matrix, float* conv_mask, float* output, int r, int width, int height);
// __global__ void conv2d_tiled(float* matrix, float* output, int width, int height);

#endif //KERNELS_H