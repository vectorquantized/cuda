
#include "cpu/cpu_kernels.h"
#include "csrc/timing_utils.h"

void matmul_cpu(const float* a, const float* b, float* c, int M, int K, int N) {
    TIMED_CPU_FUNCTION();
    for(int row=0; row < M; ++row) {
        for(int col = 0; col < N; ++col) {
            float value = 0.0f;
            for (int k = 0; k < K; ++k) {
                value += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = value;
        }
    }
}

void conv1d_cpu(const std::vector<float>& matrix, const std::vector<float>& conv_mask, std::vector<float>& output, int mask_width, int width) {
    
    TIMED_CPU_FUNCTION();
    int half_mask = mask_width / 2;

    for (int idx = 0; idx < width; ++idx) {
        float p_value = 0.0f;
        int start = idx - half_mask;
        for (int j = 0; j < mask_width; ++j) {
            if (start + j >= 0 && start + j < width) {
                p_value += matrix[start + j] * conv_mask[j];
            }
        }
        output[idx] = p_value;
    }
}

void conv2d_cpu(const std::vector<float>& matrix, const std::vector<float>& conv_mask, 
                std::vector<float>& output, int r, int width, int height) {

    TIMED_CPU_FUNCTION();
    int mask_size = 2 * r + 1;
    for(int row = 0; row < height; ++ row) {
        for (int col = 0; col < width; ++col) {
            float p_value = 0.0f;
            for (int i = 0; i < mask_size; ++i) {
                for (int j = 0; j < mask_size; ++j) {
                    int in_row = row + i - r;
                    int in_col = col + j - r;
                    if (in_row >= 0 && in_row < height &&
                        in_col >= 0 && in_col < width) {
                            p_value += matrix[in_row * width + in_col] * conv_mask[i * mask_size + j];
                        }
                }
            }
        output[row * width + col] = p_value;
        }
    }
}

#include <vector>
#include <cmath>
#include <algorithm>

// Function to apply softmax to a 2D matrix stored in a flat array in row-major order
void softmax_cpu(float* mat, int M, int N) {
    for (int row = 0; row < M; ++row) {
        float max_val = -1e20f;
        float sum = 0.0f;

        // Step 1: Find the maximum value for numerical stability
        for (int col = 0; col < N; ++col) {
            float value = mat[row * N + col];
            if (value > max_val) {
                max_val = value;
            }
        }

        // Step 2: Compute exponentials and their sum
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = std::exp(mat[row * N + col] - max_val);
            sum += mat[row * N + col];
        }

        // Step 3: Normalize the values
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] /= sum;
        }
    }
}