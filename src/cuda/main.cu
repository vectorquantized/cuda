#include <iostream>
#include <iomanip>
#include <cxxopts.hpp>
#include <unordered_map>
#include <functional>
#include "cuda/kernels.h"

// For simplicity we'll create a str -> function map and invoke the function.
std::unordered_map<std::string, std::function<void(void)>> function_map =  {
    // {
    //     "conv1d", []() { entry::conv1d_kernel_invocation(); }
    // },
    {
        "gemm", []() { gemm_kernels::launch(); }
    },
    {
        "conv2d", []() { conv2d_kernels::launch(); }
    },
    {
        "softmax", []() { softmax_kernels::launch(); }
    },
    {
        "add", []() { reduction_kernels::launch("add"); }
    },
};


// I don't want to modify main over and over again, small modifications are okay, for example the kernel_implementations vector.
// This is a good compromise, learn faster and test faster.
// The objective here is to add a kernel, add invocation, poor person's benchmark.
// More benchmarking will be added, but for now, I focus on learning.
int main(int argc, char* argv[]) {
    
    std::vector<std::string> kernel_implementations = {"conv1d", "gemm", "conv2d", "softmax"};

    cxxopts::Options options("CudaProgramming", "CLI for running cuda kernels based on learning from PMPP book.");
    options.add_options()
        ("n, name", "Kernel name", cxxopts::value<std::string>()->default_value("conv2d"), "NAME")
        ("h, help", "Print usage");
    
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    std::string func_name = result["name"].as<std::string>();

    if (function_map.find(func_name) != function_map.end()) {
        std::cout << "Running " << func_name << " kernel" << std::endl;
        function_map[func_name]();
    } else {
        std::cout << func_name << " kernel not implemented." << std::endl; 
    }
    return 0;
}