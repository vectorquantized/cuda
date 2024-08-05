
#include "cpu/cpu_kernels.h"
#include "csrc/timing_utils.h"

void matmul_cpu(const float* a, const float* b, float* c, int M, int K, int N) {
    TIME_FUNCTION();
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