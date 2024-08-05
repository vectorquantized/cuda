#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <opencv2/opencv.hpp>

namespace matrix {
// Function to compare two matrices
inline bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols, float epsilon = 1e-4) {
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(mat1[i] - mat2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}
}

namespace imageio {
inline cv::Mat readImage(std::string image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not open or find the image at " << image_path << std::endl;
        exit(EXIT_FAILURE);
    }
    return image;
}

inline cv::Mat toFloatMat(const cv::Mat& mat) {
    cv::Mat float_mat;
    if (mat.type() != CV_32F) {
        mat.convertTo(float_mat, CV_32F);
    } else {
        float_mat = mat;
    }
    return float_mat;
}

inline void matToFloatArray(const cv::Mat& mat, float* out, size_t size) {
    cv::Mat floatMat = toFloatMat(mat);
    std::memcpy(out, floatMat.data, size * sizeof(float));
}

inline cv::Mat floatArrayToMat(const float* floatArray, int width, int height, int channels) {
    cv::Mat image(height, width, CV_32FC(channels));
    std::memcpy(image.data, floatArray, width * height * channels * sizeof(float));
    return image;
}

inline void writeImage(const float* float_array, int width, int height, int num_channels, std::string output_path) {
// convert float array to matrix and write to disk.
    cv::Mat restored = imageio::floatArrayToMat(float_array, width, height, num_channels);
    bool success = cv::imwrite(output_path, restored);
    if (!success) {
        std::cerr << "Could not write image to disk at " << output_path << std::endl;
        exit(EXIT_FAILURE);
    }
}

}

#endif // UTILS_H