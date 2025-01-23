Learning to write CUDA kernels.

* This repository contains CUDA kernels from basic to somewhat optimized of some common parallel algorithmic patterns.

  ### Compile

  To run the examples, one could just do the following:

  ```
  git clone git@github.com:vectorquantized/cuda.git
  cd cuda
  mkdir build
  cd build
  apt-get update && apt-get install -y libopencv-dev
  cmake ..
  make
  ```

  The above will create a binary named `CudaProgramming`.

  ### Run

  To run a kernel, for example a `conv2d` kernel (which defaults to the tiled implementation), we do the following:

  ```
   ./CudaProgramming -n conv2d
  ```

  There's also a sample gemm.ipynb file that uses the same code used by the gemm kernel but in a jupyter notebook. It is pretty handy to verify kernel implementation when in a hurry or you don't have access to local GPU.

  ### Metrics

* To get metrics and measure performance of the kernel, use the following steps:

  * Basic Metrics:

    ```bash
    ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
    dram__bytes.sum,\
    l1tex__t_bytes.sum,\
    sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
    sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
    sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    ./CudaProgramming	
    ```

    

  * Warp and FMA Metrics:

    ```bash
    ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
    sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
    ```

  * Memory Throughput:

    ```bash
    ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    ```



