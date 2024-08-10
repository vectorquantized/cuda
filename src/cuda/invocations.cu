
// #include <iostream>
// #include <cuda_runtime.h>
// #include <iomanip>
// #include "csrc/utils.h"
// #include "invocations.h"
// #include "csrc/matrix.h"
// #include "csrc/init_utils.h"
// #include "cpu/cpu_kernels.h"
// #include "kernels.h"


// namespace entry {

// void conv1d_kernel(float* matrix_d, float* conv_mask_d, float* output_d, int mask_width, int width) {
//     TIMED_CUDA_FUNCTION();
//     int block_size = 256;
//     dim3 threads_per_block(block_size, 1, 1);
//     dim3 blocks_per_grid((width + (block_size - 1))/block_size, 1, 1);

//     conv1d<<<blocks_per_grid, threads_per_block>>>(matrix_d, conv_mask_d, output_d, mask_width, width);
//     cudaDeviceSynchronize();

// }


// void conv1d_kernel_invocation() {
//     int N = 1 << 20;
//     std::vector<float> matrix_h(N);
//     random_init(matrix_h.data(), N);
//     std::vector<float> conv_mask_h = {0.2, 0.5, 0.2};
//     int mask_width = conv_mask_h.size();
//     int width = matrix_h.size();
//     std::vector<float> cpu_output(width);

//     conv1d_cpu(matrix_h, conv_mask_h, cpu_output, mask_width, width);

//     float *matrix_d, *conv_mask_d, *output_d;

//     float* output_h = new float[width];

//     CUDA_ERROR_CHECK(cudaMalloc((void**) &matrix_d, sizeof(float) * width));
//     CUDA_ERROR_CHECK(cudaMalloc((void**) &conv_mask_d, sizeof(float) * mask_width));
//     CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, sizeof(float) * width));

//     CUDA_ERROR_CHECK(cudaMemcpy(matrix_d, matrix_h.data(), sizeof(float) * width, cudaMemcpyHostToDevice));
//     CUDA_ERROR_CHECK(cudaMemcpy(conv_mask_d, conv_mask_h.data(), sizeof(float) * mask_width, cudaMemcpyHostToDevice));

//     conv1d_kernel(matrix_d, conv_mask_d, output_d, mask_width, width);
    
//     CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, sizeof(float) * width, cudaMemcpyDeviceToHost));

//     if (utils::compare_vectors(output_h, cpu_output.data(), N)) {
//         std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
//     } else {
//         std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
//     }
    
//     delete[] output_h;
//     cudaFree(matrix_d);
//     cudaFree(conv_mask_d);
//     cudaFree(output_d);
// }