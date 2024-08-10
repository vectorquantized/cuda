#ifndef KERNELS_H
#define KERNELS_H

#define CHANNELS 3
#define TILE_WIDTH 16


#include <cuda_runtime.h>
#include "kernels/convolutions.h"

__global__ void matmul_cuda(float* a, float* b, float* c, int M, int K, int N);
__global__ void matmul_cuda_tiled(const float* __restrict__  A, const float* __restrict__ B, float* C, int M, int K, int N);
__global__ void rgbToGrayScale(float *in, float *out, int width, int height);
__global__ void saxpy_grid_strided(float a, float* b, float* c, int N);
__global__ void saxpy(float a, float* b, float* c, int N);
__global__ void conv1d(float* matrix, float* conv_mask, float* output, int mask_width, int width);


#endif //KERNELS_H