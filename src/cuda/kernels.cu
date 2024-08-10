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
