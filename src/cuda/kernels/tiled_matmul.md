# Tiled Matrix Multiplication

Tiled matrix multiplication is an efficient approach that reduces memory traffic and provides contiguous access through thread coalescing. The key concept involves reading data once from high-bandwidth memory (HBM) into shared memory, where it is reused for multiple operations, significantly improving overall performance. Here are my notes:

1. **Reuse of Shared Memory for Multiple Operations:**
    - Instead of having each output thread individually read the row and column needed for its calculation, threads in a block cooperate to load a **tile** of the matrices into shared memory. This tile is then reused across multiple operations, significantly reducing redundant memory accesses. The re-use in the example below happens while calculating the values of `P10` and `P11`, we reuse `m10` and `m11` (they were loaded once) and threads within this `block (1,1)` end up re-using the values already read from HBM.
  
2. **Efficient Use of Bandwidth:**
    - By loading data into shared memory once per tile and reusing it, the kernel reduces the number of global memory accesses. Shared memory provides much faster access than global memory, making the computation more bandwidth-efficient.

3. **Improved Data Locality:**
    - The tiling strategy improves data locality by keeping frequently accessed data (the tiles) close to the threads in shared memory. This reduces cache misses and enhances performance by minimizing the need to go back to global memory for the same data.

4. **Reduction in Global Memory Writes:**
    - Partial results are accumulated during each phase of the computation, avoiding frequent writes to global memory. Only after all phases are completed is the final result written back to global memory, reducing the write bandwidth pressure.

### How Tiling Works
1. **Phase 1:**
    - Load the first tile of the matrices (threads in a block load these in a cooperative manner) into shared memory.
    - Perform partial multiplication and accumulate the result in scalar variables.

2. **Phase 2 (and subsequent phases):**
    - Load the next tiles and continue accumulating partial results.

3. **Final Write:**
    - Once all phases are completed, the final result is written back to global memory.

![Tiled Matrix Multiplication](tiled_matmul.png)

<p align="center"><em>Toy example of tiled matmul showing the computation performed for the last output block</em></p>



<img src="matmul_tiled_odd_dims.png" alt="Tiled Matrix Multiplication" style="zoom:200%;" />



<p align="center"><em>Toy example of tiled matmul showing the computation performed for the slice [2:4, 2:4] in the output matrix</em></p>
