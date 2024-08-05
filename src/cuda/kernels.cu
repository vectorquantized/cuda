
#include "kernels.h"

#define CHANNELS 3


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

// CUDA kernel for matrix multiplication using shared memory and tiling
__global__ void matmul_cuda_tiled(float* A, float* B, float* C, int M, int K, int N) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Shared memory for tiles of A and B
    __shared__ float shared_A[16][16];
    __shared__ float shared_B[16][16];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = K * 16 * blockRow;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + K - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = 16;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = 16 * blockCol;

    // Step size used to iterate through the sub-matrices of B
    int bStep = 16 * N;

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        // Each thread loads one element of each matrix
        shared_A[row][col] = A[a + K * row + col];
        shared_B[row][col] = B[b + N * row + col];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < 16; ++k) {
            Csub += shared_A[row][k] * shared_B[k][col];
        }

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    // Each thread writes one element
    int c = N * 16 * blockRow + 16 * blockCol;
    C[c + N * row + col] = Csub;
}
