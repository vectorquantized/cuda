
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

// void matmul_kernel(float* mat1_d, float* mat2_d, float* out_d, int M, int K, int N) {
//     TIMED_CUDA_FUNCTION();
//     int block_size_x = 16;
//     int block_size_y = 16;

//     dim3 threads_per_block(block_size_x, 
//                            block_size_y);

//     dim3 blocks_per_grid((N + block_size_x - 1) / block_size_x, 
//                          (M + block_size_y - 1) / block_size_y);

//     matmul_cuda_tiled<<<blocks_per_grid, threads_per_block>>>(mat1_d, mat2_d, out_d, M, K, N);
//     cudaDeviceSynchronize();
// }

// // __global__ void conv2d_tiled(const float* __restrict__ matrix, float* output, int width, int height) {

// //     int row_output = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
// //     int col_output = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;

// //     int row_input = row_output - FILTER_RADIUS;
// //     int col_input = col_output - FILTER_RADIUS;

// //     int shared_row = threadIdx.y;
// //     int shared_col = threadIdx.x;

// //     __shared__ float matrix_shared[IN_TILE_WIDTH][IN_TILE_WIDTH];

// //     if (row_input >= 0 && row_input < height && col_input >= 0 && col_input < width) {
// //         matrix_shared[shared_row][shared_col] = __ldg(&matrix[row_input * width + col_input]);
// //     } else {
// //         matrix_shared[shared_row][shared_col] = 0.0f;
// //     }
// //     __syncthreads();
   
// //     if (shared_col < OUT_TILE_WIDTH && shared_row < OUT_TILE_WIDTH) {
// //         float p_value = 0.0f;
// //         for(int i = 0 ; i < FILTER_SIZE; ++i) {
// //             for(int j = 0; j < FILTER_SIZE; ++j) {
// //                 float mask_value = __ldcs(&Conv_Mask_C[i * FILTER_SIZE + j]);
// //                 p_value +=  mask_value * matrix_shared[shared_row + i][shared_col + j];
// //                 // p_value +=  Conv_Mask_C[i * FILTER_SIZE + j] * matrix_shared[shared_row + i][shared_col + j];
// //             }
// //         }
  
// //         if (row_output < height && col_output < width) {
// //             output[row_output * width + col_output] = p_value;
// //         }
// //     }
// // }


// // void conv2d_kernel(float* matrix_d, float* output_d, int width, int height) {
// //     TIMED_CUDA_FUNCTION();
// //     int block_size_x = IN_TILE_WIDTH;
// //     int block_size_y = IN_TILE_WIDTH;
// //     int block_dim_x = (width + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH;
// //     int block_dim_y = (height + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH;
// //     dim3 threads_per_block(block_size_x, block_size_y, 1);
// //     dim3 blocks_per_grid(block_dim_x, block_dim_y, 1);

// //     conv2d_tiled<<<blocks_per_grid, threads_per_block>>>(matrix_d, output_d, width, height);
// //     CUDA_ERROR_CHECK(cudaDeviceSynchronize());

// // }

// // void conv2d_kernel_invocation() {
// //     int M = 2048;
// //     int N = 2038;
// //     int k = FILTER_SIZE;
// //     int r = FILTER_RADIUS;

// //     std::vector<float> matrix(M * N);
// //     random_init(matrix.data(), matrix.size());
// //     std::vector<float> conv_mask(k * k);
// //     random_init(conv_mask.data(), conv_mask.size());
// //     std::vector<float> cpu_output(M * N, 0.0f);
    
// //     conv2d_cpu(matrix, conv_mask, cpu_output, r, M, N);

// //     float *matrix_d, *conv_mask_d, *output_d;
// //     int matrix_bytes = sizeof(float) * M * N;
// //     int mask_bytes = sizeof(float) * k * k;
// //     float *output_h = new float[M * N];

// //     CUDA_ERROR_CHECK(cudaMalloc((void**) &matrix_d, matrix_bytes));
// //     // CUDA_ERROR_CHECK(cudaMalloc((void**) &conv_mask_d, mask_bytes));
// //     CUDA_ERROR_CHECK(cudaMalloc((void**) &output_d, matrix_bytes));

// //     CUDA_ERROR_CHECK(cudaMemcpy(matrix_d, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
// //     // CUDA_ERROR_CHECK(cudaMemcpy(conv_mask_d, conv_mask.data(), mask_bytes, cudaMemcpyHostToDevice));
// //     CUDA_ERROR_CHECK(cudaMemcpyToSymbol(Conv_Mask_C, conv_mask.data(), mask_bytes));


// //     float hostConvMask[FILTER_SIZE * FILTER_SIZE];
// //     cudaMemcpyFromSymbol(hostConvMask, Conv_Mask_C, mask_bytes);

// //     bool is_mask_same = matrix::compare_matrices(hostConvMask, conv_mask.data(), k, k);
// //     std::cout << "is_mask_same: " << is_mask_same << std::endl;
    
// //     conv2d_kernel(matrix_d, output_d, M, N);

// //     CUDA_ERROR_CHECK(cudaMemcpy(output_h, output_d, matrix_bytes, cudaMemcpyDeviceToHost));

// //     if (matrix::compare_matrices(output_h, cpu_output.data(), M, N)) {
// //         std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
// //     } else {
// //         std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
// //     }

// //     delete[] output_h;
// //     cudaFree(matrix_d);
// //     // cudaFree(conv_mask_d);
// //     cudaFree(output_d);
// // }

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


// void matmul_invocation() {

//     int M = 2048;
//     int K = 1024;
//     int N = 2048;
//     Matrix mat1_h(M, K, random_init);
//     Matrix mat2_h(K, N, random_init);
//     float* out_h = new float[M * N];
//     float* out_cpu = new float[M * N];
//     matmul_cpu(mat1_h.data.get(), mat2_h.data.get(), out_cpu, M, K, N);
//     float *mat1_d, *mat2_d, *out_d;

//     CUDA_ERROR_CHECK(cudaMalloc((void**) &mat1_d, M * K * sizeof(float)));
//     CUDA_ERROR_CHECK(cudaMalloc((void**) &mat2_d, K * N * sizeof(float)));
//     CUDA_ERROR_CHECK(cudaMalloc((void**) &out_d, M * N * sizeof(float)));

//     CUDA_ERROR_CHECK(cudaMemcpy(mat1_d, mat1_h.data.get(), M * K * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_ERROR_CHECK(cudaMemcpy(mat2_d, mat2_h.data.get(), K * N * sizeof(float), cudaMemcpyHostToDevice));

//     matmul_kernel(mat1_d, mat2_d, out_d, M, K, N);
    
//     CUDA_ERROR_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

//     if (matrix::compare_matrices(out_h, out_cpu, M, N)) {
//         std::cout << "The CUDA kernel's result matches the CPU result." << std::endl;
//     } else {
//         std::cerr << "The CUDA kernel's result does NOT match the CPU result." << std::endl;
//     }

//     delete [] out_h;
//     delete [] out_cpu;
//     cudaFree(mat1_d);
//     cudaFree(mat2_d);
//     cudaFree(out_d);
// }
// }
