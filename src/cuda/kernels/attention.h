#ifndef ATTENTION_H
#define ATTENTION_H

#include <cuda_runtime.h>


__global__ void fused_attention(const float* q, const float* k, const float* v, float* output, int M, int D_model, int N);

namespace attn {
void launch();
};


#endif // ATTENTION_H
