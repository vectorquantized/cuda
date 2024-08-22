# Warps, SIMD Hardware and Occupancy

Within a block, threads can execute in any order relative to each other. In algorithms with phases, barrier synchronization ensures that all threads finish one phase before starting the next. A good example is the tiled matrix multiplication implementation. Without synchronization, no specific ordering of threads is guaranteed within a block. Thread scheduling is handled at the hardware level. Once a block is assigned to an SM, it’s further split into thread groups called warps. Warp size is architecture-dependent and serves as the basic unit of thread scheduling in an SM.

An SM is designed to execute all threads within a warp under the Single Instruction, Multiple Data (SIMD) model. Due to SIMD hardware constraints, all threads in a warp must follow the same instruction at any point, often referred to as Single Instruction Multiple Threads (SIMT). In SIMD hardware, control logic costs are spread across many execution units, meaning only a small fraction of hardware is used for control, leaving the majority for arithmetic throughput.

### Control Divergence

SIMD execution performs well when all threads in a warp take the same execution path. For an `if-else` construct, this works well if all threads either take the `if` path or the `else` path. However, if there’s divergence—where some threads take the `if` path and others take the `else` path—the SIMD hardware will make multiple passes through these paths. During each pass, threads not on that path are masked out.

On Pascal and earlier architectures, such passes were executed sequentially. Starting with Volta, these passes can be interleaved and executed in parallel (known as Independent Thread Scheduling), improving efficiency when divergence occurs.

### Warp Scheduling and Latency Tolerance

Typically, more threads are assigned to an SM than it can run concurrently. This allows for better utilization when some threads are stalled, such as during long-latency operations like global memory access. Other threads can be scheduled to fill this gap, a technique called latency tolerance. If a warp must wait for a previously initiated long-latency operation, it’s not selected for execution, and another warp that’s ready is scheduled instead.

This selection process avoids idle time, leading to zero-overhead thread scheduling. This capability for handling long latencies is why GPUs don’t allocate much chip area to branch prediction or cache memories, unlike CPUs.

### Occupancy

Occupancy is the ratio of the number of warps assigned to an SM to the maximum possible warps. A 100% occupancy means that all available slots for warps are filled, which theoretically maximizes the utilization of available resources

**Example Calculation:**

For the H100:

- Total SMs: 144
- Blocks per SM: 32
- Max Threads per Block: 1024
- Warps per SM: 64 ⟶ 2048 threads per SM (since warp size for H100 is 32)

100% occupancy can be achieved through configurations between two extremes:

- 1024 threads per block ⟶ 2 blocks per SM, totaling 2048 threads.
- 64 threads per block ⟶ 32 blocks per SM, totaling 2048 threads.

If threads per block drop below 32, achieving max occupancy requires 64 blocks per SM ($\frac{2048}{32}$). But with only 32 blocks allowed per SM, total threads would be $32 \times 32 = 1024$, resulting in 50% occupancy ($\frac{1024}{2048}$). Thus, to optimize kernel performance, aim for at least 64 threads per block.

Another factor is aligning the total threads with threads per block. For instance, with 768 threads per block, only 2 blocks fit per SM, totaling 1536 threads, wasting 512 threads and reducing occupancy to 75% ($\frac{1536}{2048}$). You might also need the total thread count to be a multiple of warp size.

Register availability also impacts occupancy. On the H100:

- Total Registers per SM: 64K (65536)
- Max Registers per Thread: 255

To maintain 100% occupancy with 2048 threads per SM, each thread should use no more than $\frac{65536}{2048} = 32$ registers. If a kernel uses more registers per thread, occupancy decreases since fewer threads can be launched per block.

**Example:**

A kernel using 34 registers per thread, launched with 512 threads per block ($\frac{2048}{512} = 4$ blocks), would require $34 \times 2048 = 69632$ registers per SM. The CUDA runtime might only launch 3 blocks per SM instead, resulting in $512 \times 3 \times 34 = 52224$ registers but with an occupancy of 75% ($\frac{512 \times 3}{2048}$​). This drop in parallelism due to using slightly more registers is known as a performance cliff.

Although, we focussed on 100% occupancy here but that doesn't always  guarantee improved performance. Sometimes, reducing occupancy by using more registers can result in better performance due to memory coalescing, reduced contention, or even lower instruction divergence.

**Memory Coalescing**: With more registers, threads can hold onto data longer, potentially allowing for more efficient coalesced memory access patterns when they do need to access global memory.

**Reduced Contention**: Fewer active threads mean less contention for shared resources like memory bandwidth.

We shall see the above perhaps when I implement the shared memory and regsiter tiled version of matmul.

---

## References

* **Chapter 4 (Compute Architecture and Scheduling):** 
  *Programming Massively Parallel Processors* 
  Wen-mei W. Hwu