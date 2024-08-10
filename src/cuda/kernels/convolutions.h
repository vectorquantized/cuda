
#ifndef CONVOLUTIONS_H
#define CONVOLUTIONS_H

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * (FILTER_RADIUS) + 1)
#define OUT_TILE_WIDTH 12
#define IN_TILE_WIDTH ((OUT_TILE_WIDTH) + (FILTER_SIZE) - 1)

#include <cuda_runtime.h>


__global__ void conv2d(float* matrix, float* conv_mask, float* output, int r, int width, int height);
__global__ void conv2d_tiled(const float* __restrict__ matrix, float* output, int width, int height);

namespace conv2d_kernels {
void launch();
};


#endif // CONVOLUTIONS_H