#include <math.h>
#include <cuda_runtime.h>

#include "kernels.h"
#include "csrc/utils.h"


int main_grayscale() {

    std::string image_path = "/root/Macaw.jpg";
    std::string output_path = "/root/Macaw_grayscale.jpg";
    cv::Mat image_mat = imageio::readImage(image_path);
    int width = image_mat.cols;
    int height = image_mat.rows;
    int channels = image_mat.channels();
    int size_colored = width * height * channels;
    int size_grayscale = width * height;
    float *colored_image_h = new float[size_colored];
    float *gray_scale_D, *colored_image_D;

    // matrix to float array
    imageio::matToFloatArray(image_mat, colored_image_h, size_colored);
    
    // allocate memory on device
    cudaMalloc((void**) &colored_image_D, size_colored * sizeof(float));
    cudaMemcpy(colored_image_D, colored_image_h, size_colored * sizeof(float), cudaMemcpyHostToDevice);

    //allocate memory for output in unified memory
    cudaMallocManaged(&gray_scale_D, size_grayscale * sizeof(float));

    // kernel launch params.
    int block_size = 16;
    int num_blocks_x = (width + block_size - 1) / block_size;
    int num_blocks_y = (height + block_size - 1) / block_size;

    dim3 dim_grid(num_blocks_x, num_blocks_y, 1);
    dim3 dim_block(block_size, block_size, 1);

    // launch kernel
    rgbToGrayScale<<<dim_grid, dim_block>>>(colored_image_D, gray_scale_D, width, height);
    //synchronize
    cudaDeviceSynchronize();

    imageio::writeImage(gray_scale_D, width, height, 1, output_path);

    // free memory
    delete[] colored_image_h;
    cudaFree(colored_image_D);
    cudaFree(gray_scale_D);
    return 0;
}