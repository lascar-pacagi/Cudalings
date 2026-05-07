// gpt2_backward.cu -- backward kernels for the gpt2_forward in chapter 23.
//
// Skeleton -- each kernel has the right signature and a comment explaining
// the math. Filling these in is the work of chapter 24, supported by the
// progressive CUDAlings exercises in Part8/exercises/24_llm_c_bwd/.
//
// Compile alongside gpt2_forward.cu when both files exist:
//   nvcc -O2 -arch=sm_61 -std=c++17 gpt2_backward.cu -c -o gpt2_backward.o

#include <cstdio>
#include <cuda_runtime.h>


// ===========================================================================
// matmul_backward: given dy = (M, OC), produce dx = (M, IC) and dW = (OC, IC).
//   dx = dy · W            (M, IC) = (M, OC) · (OC, IC)
//   dW = dy^T · x          (OC, IC) = (OC, M) · (M, IC)
// ===========================================================================
__global__ void matmul_backward_dx_kernel(
    float* dx, const float* dy, const float* W,
    int M, int IC, int OC)
{
    // TODO (chapter 24): tile-reduce dy (M, OC) @ W (OC, IC) → dx (M, IC).
    // Same tiling pattern as forward matmul, just different shapes.
    (void)dx; (void)dy; (void)W; (void)M; (void)IC; (void)OC;
}

__global__ void matmul_backward_dW_kernel(
    float* dW, const float* dy, const float* x,
    int M, int IC, int OC)
{
    // TODO: dW[oc, ic] = sum_m dy[m, oc] * x[m, ic]
    // Use one block per (oc, ic) tile, reduce across m.
    (void)dW; (void)dy; (void)x; (void)M; (void)IC; (void)OC;
}


// ===========================================================================
// layernorm_backward: given dy and saved (mean, rstd), compute dx, dgamma, dbeta.
// Uses the compact form from llm.c:
//   dx = (γ * rstd) · (dy - mean(dy_hat) - x_hat * mean(dy_hat * x_hat))
// ===========================================================================
__global__ void layernorm_backward_kernel(
    float* dx, float* dgamma, float* dbeta,
    const float* dy, const float* x,
    const float* mean, const float* rstd, const float* gamma,
    int B, int T, int E)
{
    // TODO (chapter 24): one block per (b, t). See llm.c/layernorm_backward.
    (void)dx; (void)dgamma; (void)dbeta;
    (void)dy; (void)x; (void)mean; (void)rstd; (void)gamma;
    (void)B; (void)T; (void)E;
}


// ===========================================================================
// gelu_backward: dx = dy * d/dx(gelu(x)).
// For tanh-approx GELU, the derivative has a closed form -- copy it from
// the chapter README's cheatsheet.
// ===========================================================================
__global__ void gelu_backward_kernel(
    float* dx, const float* dy, const float* x, int n)
{
    // TODO: write the closed form.
    (void)dx; (void)dy; (void)x; (void)n;
}


// ===========================================================================
// attention_backward: composes softmax_backward + matmul_backward correctly.
// Hardest kernel in the chapter. See exercises 24.4 and 24.5.
// ===========================================================================
__global__ void attention_backward_kernel(
    float* dqkv, const float* datt_unused,
    const float* qkv, const float* att, const float* dout,
    int B, int T, int E, int H)
{
    // TODO: see exercises 24.4-24.5. Returns dqkv[b,t,3*E].
    (void)dqkv; (void)datt_unused;
    (void)qkv; (void)att; (void)dout;
    (void)B; (void)T; (void)E; (void)H;
}


// ===========================================================================
// encoder_backward: scatter-add dy into dwte (atomic) and dwpe.
// ===========================================================================
__global__ void encoder_backward_kernel(
    float* dwte, float* dwpe,
    const float* dy, const int* idx,
    int B, int T, int E)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int b = blockIdx.z;
    if (e >= E) return;
    float g = dy[(b * T + t) * E + e];
    int token = idx[b * T + t];
    atomicAdd(&dwte[token * E + e], g);   // multiple (b,t) may share token
    atomicAdd(&dwpe[t     * E + e], g);   // and (b) all share position t
}


// ===========================================================================
// crossentropy_softmax_backward: combined softmax + cross-entropy gradient.
// dlogits[i] = (softmax(logits)[i] - one_hot(target)[i]) / N
// Numerically stable because we never materialize log(softmax).
// ===========================================================================
__global__ void crossentropy_softmax_backward_kernel(
    float* dlogits, const float* probs, const int* targets,
    int B, int T, int V, float scale)
{
    // TODO: see exercise 24.6.
    (void)dlogits; (void)probs; (void)targets;
    (void)B; (void)T; (void)V; (void)scale;
}


int main() {
    printf("gpt2_backward.cu: skeleton compiled. "
           "Fill in the TODOs by working through Part8/exercises/24_llm_c_bwd/.\n");
    return 0;
}
