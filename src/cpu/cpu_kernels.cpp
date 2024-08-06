
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