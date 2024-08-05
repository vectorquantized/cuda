
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(call)  { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i.%s \n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }}

#endif // CUDA_UTILS_H


