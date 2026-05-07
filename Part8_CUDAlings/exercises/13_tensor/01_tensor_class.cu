// CUDAlings 13.01 — A minimal Tensor RAII wrapper
//
// Goal: complete Tensor::Tensor(size_t n) and Tensor::~Tensor() so that the
// constructor cudaMallocs and the destructor cudaFrees. Tensor must own a
// device pointer and a size. Then the test below should print "ok".

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

struct Tensor {
    float* data = nullptr;
    size_t n = 0;

    Tensor() = default;
    Tensor(size_t n_) : n(n_) {
        // TODO: allocate `data` for `n` floats on the device
    }
    ~Tensor() {
        // TODO: release the device allocation if it isn't null
    }
    // Disable copy; allow move (a Tensor uniquely owns its memory).
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept : data(o.data), n(o.n) { o.data = nullptr; o.n = 0; }
};

__global__ void fill(float* p, float v, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main() {
    {
        Tensor t(1024);
        if (!t.data) { printf("FAIL: data is null\n"); return 1; }
        fill<<<(int)((t.n + 255)/256), 256>>>(t.data, 1.0f, (int)t.n);
        cudaDeviceSynchronize();
    }   // <- destructor must free without error
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
    printf("ok\n");
    return 0;
}
