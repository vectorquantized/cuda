#ifndef REDUCTIONS_H
#define REDUCTIONS_H

#include <string>

__global__ void add(float* input, float* output, int M);

namespace reduction_kernels {
    void launch(std::string name);
}

#endif // REDUCTIONS_H