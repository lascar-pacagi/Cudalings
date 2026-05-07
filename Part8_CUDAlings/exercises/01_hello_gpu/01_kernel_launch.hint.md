# Hint — 01.01 kernel_launch

- The triple-angle launch syntax is `kernel<<<gridDim, blockDim>>>(args)`.
- `threadIdx.x` is a built-in that's only valid inside a `__global__` or
  `__device__` function.
- After the launch, **always** call `cudaDeviceSynchronize()` if you want
  to see device printf output.
