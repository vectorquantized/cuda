Learning to write CUDA kernels.

* This repository contains CUDA kernels from basic to somewhat optimized of some common parallel algorithmic patterns.
* To run the examples, one could just do the following:
  ```
  git clone git@github.com:vectorquantized/cuda.git
  cd cuda
  mkdir build
  cd build
  cmake ..
  make
  ```
* The above will create a binary named `CudaProgramming`.
* To run a kernel, for example a `conv2d` kernel (which defaults to the tiled implementation), we do the following:
  ```
   ./CudaProgramming -n conv2d
  ```
* There's also a sample gemm.ipynb file that uses the same code used by the gemm kernel but in a jupyter notebook. It is pretty handy to verify kernel implementation when in a hurry or you don't have access to local GPU.
  
