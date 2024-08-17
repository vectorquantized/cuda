
#include <iostream>
#include <vector>
#include <iomanip>
#include "csrc/utils.h"
#include "csrc/matrix.h"
#include "csrc/init_utils.h"
#include "cpu/cpu_kernels.h"
#include "cuda/utils/cuda_utils.h"
#include "attention.h"



namespace attn {
void launch() {

    int B = 16; //batch size
    int S = 16; // sequence length
    int D_model = 8; // dim of Q, K and V
    
    //initialize host q, k, v
    // use 3d tensor implementation. 
    CUDA_ERROR_CHECK(cudaMalloc((void**) &q_d, B * S * D_model * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &k_d, B * S * D_model * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &v_d, B * S * D_model * sizeof(float)));

    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
}
}
