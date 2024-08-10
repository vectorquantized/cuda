#ifndef GEMM_H
#define GEMM_H

#define TILE_WIDTH 16

#include <cuda_runtime.h>


__global__ void gemm_cuda(const float* __restrict__ a, const float* __restrict__ b, float* c, int M, int K, int N);
__global__ void gemm_cuda_tiled(const float* __restrict__  A, const float* __restrict__ B, float* C, int M, int K, int N);

namespace gemm_kernels {
void launch();
};


#endif // GEMM_H