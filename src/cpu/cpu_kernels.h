#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include <vector>

void matmul_cpu(const float* a, const float* b, float* c, int M, int K, int N);
void conv1d_cpu(const std::vector<float>& matrix, const std::vector<float>& conv_mask, std::vector<float>& output, int mask_width, int width);


#endif // CPU_KERNELS_H