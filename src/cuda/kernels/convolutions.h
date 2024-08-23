
#ifndef CONVOLUTIONS_H
#define CONVOLUTIONS_H

// In tiled convolution kernels, the design gets complicated as the input tile
// width and output tile width are different.
// input tile width has 2 * filter radius more elements than output tile widths.
// In the calculations below: 
// for filter radius 2, we have a filter size of 5.
// for input tile width of 16, we'd have an output tile width of 12.
// for kernel launch, we'd launch in_tile_width threads in a block, 
// as shared memory would load these many elements in the each dimension and 
// CEIL_DIV(width, out_tile_width) blocks in x direction
// CEIL_DIV(height, out_tile_width) blocks in y direction
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
