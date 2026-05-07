// CUDAlings 25.02 — Fused softmax + cross-entropy gradient kernel
//
// At the lm_head, instead of (1) softmax to probs and (2) backward through
// softmax + xent separately, you can compute dlogits = (probs - one_hot) / N
// in one pass. This is the kernel every LLM trainer launches at the head.
//
// Forward (just for verification):  loss = -log(softmax(z)[target])
// Backward (this kernel):           dlogits[i] = (softmax(z)[i] - 1{i==target}) / N
//
// Goal: implement on a single row (B=1, V=4) with target=2. The expected
// dlogits sum to 0 (gradient through softmax always sums to 0).

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define V 4

__global__ void softmax_xent_bwd(const float* logits, int target, float* dlogits, float scale) {
    __shared__ float buf[V];
    __shared__ float row_max, row_sum;
    int i = threadIdx.x;
    if (i >= V) return;

    buf[i] = logits[i];
    __syncthreads();
    if (i == 0) {
        float m = buf[0];
        for (int k = 1; k < V; ++k) if (buf[k] > m) m = buf[k];
        row_max = m;
    }
    __syncthreads();
    float e = expf(buf[i] - row_max);
    buf[i] = e;
    __syncthreads();
    if (i == 0) {
        float s = 0;
        for (int k = 0; k < V; ++k) s += buf[k];
        row_sum = s;
    }
    __syncthreads();
    float p = e / row_sum;
    // TODO: write the (probs - one_hot[target]) * scale formula from the header into dlogits[i]
}

int main() {
    float h_logits[V] = {1.0f, 2.0f, 3.0f, 4.0f};   // any row
    int target = 2;
    float h_dl[V] = {0};
    float *d_l, *d_dl;
    cudaMalloc(&d_l, V*sizeof(float));
    cudaMalloc(&d_dl, V*sizeof(float));
    cudaMemcpy(d_l, h_logits, V*sizeof(float), cudaMemcpyHostToDevice);
    softmax_xent_bwd<<<1, V>>>(d_l, target, d_dl, /*scale*/ 1.0f);
    cudaMemcpy(h_dl, d_dl, V*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += h_dl[i];
    printf("sum=%.4f\n", sum);   // expected 0 (probs sum to 1, one_hot sums to 1)
    cudaFree(d_l); cudaFree(d_dl);
    return 0;
}
