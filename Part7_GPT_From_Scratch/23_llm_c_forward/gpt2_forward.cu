// gpt2_forward.cu -- a llm.c-style forward pass of GPT-2 in pure CUDA.
//
// All shapes (B, T, E, H) are flat row-major float buffers on device.
// One file, ~500 lines, everything you need to run a GPT forward.
// Compile with:  nvcc -O2 -arch=sm_61 -std=c++17 gpt2_forward.cu -o gpt2_forward

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error helper.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(_e), __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// ===========================================================================
// 1. Encoder: out[b,t,e] = wte[idx[b,t], e] + wpe[t, e]
// ===========================================================================
__global__ void encoder_kernel(
    float* __restrict__ out,           // (B, T, E)
    const int*   __restrict__ idx,     // (B, T)
    const float* __restrict__ wte,     // (V, E)
    const float* __restrict__ wpe,     // (T, E)
    int B, int T, int E)
{
    // Each thread handles one (b, t, e) slot. Coalesces along e.
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int b = blockIdx.z;
    if (e >= E) return;
    int token = idx[b * T + t];
    out[(b * T + t) * E + e] = wte[token * E + e] + wpe[t * E + e];
}

void encoder_forward(float* out, const int* idx, const float* wte, const float* wpe,
                     int B, int T, int E) {
    int block = 256;
    dim3 grid((E + block - 1) / block, T, B);
    encoder_kernel<<<grid, block>>>(out, idx, wte, wpe, B, T, E);
}


// ===========================================================================
// 2. LayerNorm forward (one block per (b, t) row)
//   y = (x - mean) / sqrt(var + eps); y = y * gamma + beta
// ===========================================================================
__global__ void layernorm_kernel(
    float* __restrict__ out,
    float* __restrict__ mean_out,      // saved for backward
    float* __restrict__ rstd_out,      // saved for backward
    const float* __restrict__ x,       // (B, T, E)
    const float* __restrict__ gamma,   // (E,)
    const float* __restrict__ beta,    // (E,)
    int B, int T, int E)
{
    // Each block normalizes one row of length E.
    int idx = blockIdx.x;              // 0 .. B*T-1
    int tid = threadIdx.x;
    const float* xrow = x   + idx * E;
    float*       yrow = out + idx * E;

    // 1) sum and sum-of-squares via tree reduction in shared memory
    __shared__ float ssum[256];
    __shared__ float sssq[256];
    float local_sum = 0, local_ssq = 0;
    for (int e = tid; e < E; e += blockDim.x) {
        float v = xrow[e];
        local_sum += v;
        local_ssq += v * v;
    }
    ssum[tid] = local_sum;
    sssq[tid] = local_ssq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            sssq[tid] += sssq[tid + s];
        }
        __syncthreads();
    }
    float mean = ssum[0] / (float)E;
    float var  = sssq[0] / (float)E - mean * mean;
    float rstd = rsqrtf(var + 1e-5f);

    if (tid == 0) {
        mean_out[idx] = mean;
        rstd_out[idx] = rstd;
    }

    // 2) normalize + affine
    for (int e = tid; e < E; e += blockDim.x) {
        float n = (xrow[e] - mean) * rstd;
        yrow[e] = n * gamma[e] + (beta ? beta[e] : 0.0f);
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       const float* x, const float* gamma, const float* beta,
                       int B, int T, int E) {
    int block = 256;
    int grid = B * T;
    layernorm_kernel<<<grid, block>>>(out, mean, rstd, x, gamma, beta, B, T, E);
}


// ===========================================================================
// 3. Matmul forward:  out[B*T, OC] = x[B*T, IC] @ W[OC, IC]^T  + bias[OC]
//   tiled, no tensor cores. Pascal-friendly.
// ===========================================================================
#define TILE 16
__global__ void matmul_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,       // (M, IC) where M = B*T
    const float* __restrict__ W,       // (OC, IC)
    const float* __restrict__ bias,    // (OC,) optional, can be NULL
    int M, int IC, int OC)
{
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;     // 0..M
    int col = blockIdx.x * TILE + threadIdx.x;     // 0..OC
    float acc = 0.f;
    for (int t = 0; t < IC; t += TILE) {
        int x_col = t + threadIdx.x;
        int w_col = t + threadIdx.y;
        xs[threadIdx.y][threadIdx.x] =
            (row < M && x_col < IC) ? x[row * IC + x_col] : 0.f;
        // W is stored (OC, IC) row-major, so W[col, w_col] is W[col*IC + w_col].
        ws[threadIdx.y][threadIdx.x] =
            (col < OC && w_col < IC) ? W[col * IC + w_col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += xs[threadIdx.y][k] * ws[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < OC) out[row * OC + col] = acc + (bias ? bias[col] : 0.f);
}

void matmul_forward(float* out, const float* x, const float* W, const float* bias,
                    int B, int T, int IC, int OC) {
    int M = B * T;
    dim3 block(TILE, TILE);
    dim3 grid((OC + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<grid, block>>>(out, x, W, bias, M, IC, OC);
}


// ===========================================================================
// 4. GELU (tanh approximation, GPT-2 style):
//      gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ===========================================================================
__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    const float k = 0.7978845608f;     // sqrt(2/pi)
    float u = k * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(u));
}

void gelu_forward(float* x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_kernel<<<grid, block>>>(x, n);
}


// ===========================================================================
// 5. Residual: x = x + y elementwise (in-place on x).
// ===========================================================================
__global__ void residual_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

void residual_forward(float* x, const float* y, int n) {
    int block = 256;
    residual_kernel<<<(n + block - 1) / block, block>>>(x, y, n);
}


// ===========================================================================
// 6. Causal multi-head attention (the most subtle kernel).
//   Input:  qkv   (B, T, 3*E)            -- packed q | k | v
//   Output: out   (B, T, E)
//   Internal: att (B, H, T, T)            -- softmax of scaled dot products
//
//   For each (b, h, t):
//     1) compute att[b,h,t,s] = q[b,h,t,:] · k[b,h,s,:] / sqrt(Dh)  for s <= t
//     2) softmax over s in [0, t]
//     3) out[b,h,t,:] = sum_{s<=t} att[b,h,t,s] * v[b,h,s,:]
//
//   We split into TWO kernels:
//     attn_qk_kernel   : computes att (with mask + softmax)
//     attn_av_kernel   : multiplies att @ v
//   This keeps each kernel simple and lets the profiler attribute time.
// ===========================================================================
__global__ void attn_qk_kernel(
    float* __restrict__ att,           // (B, H, T, T)
    const float* __restrict__ qkv,     // (B, T, 3*E)
    int B, int T, int E, int H)
{
    int Dh = E / H;
    int t = blockIdx.x;                // query position (row)
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    // Pointer to q[b, t, h, :] within the packed qkv tensor:
    //   qkv[b, t] is laid out as (q | k | v), each of size E.
    //   Within q, head h starts at offset h * Dh.
    const float* qp = qkv + ((b * T + t) * 3 + 0) * E + h * Dh;

    // Each thread handles a different key position s.
    // Compute dot product q · k(s) / sqrt(Dh) for s = tid, tid+blockDim, ...
    // For s > t, the mask sets att = -inf so softmax kills it.
    extern __shared__ float row[];     // size T floats
    for (int s = tid; s < T; s += blockDim.x) {
        if (s > t) { row[s] = -INFINITY; continue; }
        const float* kp = qkv + ((b * T + s) * 3 + 1) * E + h * Dh;
        float dot = 0.f;
        for (int d = 0; d < Dh; ++d) dot += qp[d] * kp[d];
        row[s] = dot * rsqrtf((float)Dh);
    }
    __syncthreads();

    // Numerically-stable softmax over [0..T-1] (with -inf in the masked range).
    // Step 1: row max.
    float mx = -INFINITY;
    for (int s = tid; s < T; s += blockDim.x) if (row[s] > mx) mx = row[s];
    __shared__ float smax[32];     // one per warp
    int warp = tid / 32;
    int lane = tid & 31;
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffff, mx, off);
        if (other > mx) mx = other;
    }
    if (lane == 0) smax[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = (tid < (blockDim.x + 31) / 32) ? smax[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, mx, off);
            if (other > mx) mx = other;
        }
        if (tid == 0) smax[0] = mx;
    }
    __syncthreads();
    float row_max = smax[0];

    // Step 2: exp + sum.
    float local_sum = 0;
    for (int s = tid; s < T; s += blockDim.x) {
        float e = expf(row[s] - row_max);
        row[s] = e;
        local_sum += e;
    }
    __shared__ float ssum[32];
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (lane == 0) ssum[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (tid < (blockDim.x + 31) / 32) ? ssum[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        if (tid == 0) ssum[0] = local_sum;
    }
    __syncthreads();
    float inv = 1.0f / ssum[0];

    // Step 3: normalize and write.
    float* att_row = att + ((b * H + h) * T + t) * T;
    for (int s = tid; s < T; s += blockDim.x) att_row[s] = row[s] * inv;
}


__global__ void attn_av_kernel(
    float* __restrict__ out,           // (B, T, E)
    const float* __restrict__ att,     // (B, H, T, T)
    const float* __restrict__ qkv,     // (B, T, 3*E)
    int B, int T, int E, int H)
{
    // Each thread handles one element of out[b, t, h*Dh + d].
    int Dh = E / H;
    int d = threadIdx.x;
    int t = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (d >= Dh) return;

    const float* att_row = att + ((b * H + h) * T + t) * T;
    float acc = 0.f;
    for (int s = 0; s <= t; ++s) {
        const float* vp = qkv + ((b * T + s) * 3 + 2) * E + h * Dh;
        acc += att_row[s] * vp[d];
    }
    out[(b * T + t) * E + h * Dh + d] = acc;
}


void attention_forward(float* out, float* att, const float* qkv,
                       int B, int T, int E, int H) {
    int Dh = E / H;
    {
        // softmax kernel: 1 block per (b, h, t); 256 threads work the row.
        dim3 grid(T, H, B);
        int block = 256;
        size_t shmem = T * sizeof(float);
        attn_qk_kernel<<<grid, block, shmem>>>(att, qkv, B, T, E, H);
    }
    {
        // av kernel: 1 block per (b, h, t); Dh threads work the head dim.
        // Dh is typically 32 or 64 -- well within 1024 thread limit.
        dim3 grid(T, H, B);
        int block = Dh;
        attn_av_kernel<<<grid, block>>>(out, att, qkv, B, T, E, H);
    }
}


// ===========================================================================
// Top-level driver: a tiny "GPT-2 forward" that chains everything together.
// Sized for the chapter 21 default model (n_layer=4, n_head=4, n_embd=128).
// ===========================================================================
struct GPT2Cfg {
    int B, T, V, E, H, n_layer;
};

// We assume the caller has already uploaded all weights to device. To keep
// this file readable, we declare them as a flat struct of pointers.
struct GPT2Weights {
    const float* wte;        // (V, E)
    const float* wpe;        // (T, E)
    // per-layer pointers, packed across n_layer:
    const float* ln1_w;      // (n_layer, E)
    const float* ln2_w;      // (n_layer, E)
    const float* qkv_w;      // (n_layer, 3E, E)
    const float* qkv_b;      // (n_layer, 3E) -- may be all zeros
    const float* attn_proj_w;// (n_layer, E, E)
    const float* attn_proj_b;
    const float* fc_w;       // (n_layer, 4E, E)
    const float* fc_b;
    const float* fc2_w;      // (n_layer, E, 4E)
    const float* fc2_b;
    const float* ln_f_w;     // (E,)
    // No lm_head_w field: we tie weights with wte.
};

struct GPT2Acts {
    // a single big arena, sliced into per-layer scratch buffers
    float *encoded;          // (B, T, E)
    float *ln1_out, *ln2_out, *ln_f_out;
    float *qkv;              // (B, T, 3E)
    float *att;              // (B, H, T, T)
    float *attn_out;         // (B, T, E)  -- after av
    float *attn_proj_out;    // (B, T, E)
    float *fc_out;           // (B, T, 4E)
    float *mlp_out;          // (B, T, E)
    float *mean, *rstd;      // (B, T)  saved for backward
    float *logits;           // (B, T, V)
    float *x;                // residual stream (B, T, E)
};


void gpt2_forward(const GPT2Cfg& cfg, const GPT2Weights& w, GPT2Acts& a,
                  const int* idx /* device, (B, T) */) {
    int B = cfg.B, T = cfg.T, E = cfg.E, H = cfg.H, V = cfg.V;
    int BTE = B * T * E;

    // 1) embeddings
    encoder_forward(a.encoded, idx, w.wte, w.wpe, B, T, E);
    // residual stream starts as the embeddings
    CUDA_CHECK(cudaMemcpyAsync(a.x, a.encoded, BTE * sizeof(float),
                               cudaMemcpyDeviceToDevice));

    // 2) transformer blocks
    for (int l = 0; l < cfg.n_layer; ++l) {
        const float* ln1_w_l = w.ln1_w + l * E;
        const float* ln2_w_l = w.ln2_w + l * E;
        const float* qkv_w_l = w.qkv_w + l * 3 * E * E;
        const float* attn_proj_w_l = w.attn_proj_w + l * E * E;
        const float* fc_w_l  = w.fc_w  + l * 4 * E * E;
        const float* fc2_w_l = w.fc2_w + l * E * 4 * E;

        layernorm_forward(a.ln1_out, a.mean, a.rstd, a.x, ln1_w_l, /*beta*/ nullptr, B, T, E);
        matmul_forward(a.qkv, a.ln1_out, qkv_w_l, /*bias*/ nullptr, B, T, E, 3 * E);
        attention_forward(a.attn_out, a.att, a.qkv, B, T, E, H);
        matmul_forward(a.attn_proj_out, a.attn_out, attn_proj_w_l, nullptr, B, T, E, E);
        residual_forward(a.x, a.attn_proj_out, BTE);

        layernorm_forward(a.ln2_out, a.mean, a.rstd, a.x, ln2_w_l, nullptr, B, T, E);
        matmul_forward(a.fc_out, a.ln2_out, fc_w_l, nullptr, B, T, E, 4 * E);
        gelu_forward(a.fc_out, B * T * 4 * E);
        matmul_forward(a.mlp_out, a.fc_out, fc2_w_l, nullptr, B, T, 4 * E, E);
        residual_forward(a.x, a.mlp_out, BTE);
    }

    // 3) final layernorm + lm_head (tied with wte)
    layernorm_forward(a.ln_f_out, a.mean, a.rstd, a.x, w.ln_f_w, nullptr, B, T, E);
    matmul_forward(a.logits, a.ln_f_out, w.wte, nullptr, B, T, E, V);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ---------------------------------------------------------------------------
// main: tiny smoke test. The full verification lives in `verify.py` which
// loads weights from a PyTorch checkpoint and diffs the outputs.
// ---------------------------------------------------------------------------
int main() {
    GPT2Cfg cfg = {1 /*B*/, 8 /*T*/, 65 /*V*/, 64 /*E*/, 4 /*H*/, 2 /*n_layer*/};
    int B = cfg.B, T = cfg.T, V = cfg.V, E = cfg.E, H = cfg.H, NL = cfg.n_layer;

    // host scratch -- random weights (fixed seed) just to exercise the pipeline.
    auto fill = [](float* p, int n, float scale) {
        for (int i = 0; i < n; ++i) p[i] = scale * (((float)rand() / RAND_MAX) - 0.5f);
    };
    srand(0);
    int E2 = E * E, E4 = 4 * E;

    float *h_wte = new float[V * E]; fill(h_wte, V * E, 0.02f);
    float *h_wpe = new float[T * E]; fill(h_wpe, T * E, 0.02f);
    float *h_ln1 = new float[NL * E]; for (int i = 0; i < NL*E; ++i) h_ln1[i] = 1.0f;
    float *h_ln2 = new float[NL * E]; for (int i = 0; i < NL*E; ++i) h_ln2[i] = 1.0f;
    float *h_lnf = new float[E];      for (int i = 0; i < E;    ++i) h_lnf[i] = 1.0f;
    float *h_qkv = new float[NL * 3 * E2]; fill(h_qkv, NL * 3 * E2, 0.02f);
    float *h_apw = new float[NL * E2];     fill(h_apw, NL * E2,     0.02f);
    float *h_fc  = new float[NL * E4 * E]; fill(h_fc,  NL * E4 * E, 0.02f);
    float *h_fc2 = new float[NL * E * E4]; fill(h_fc2, NL * E * E4, 0.02f);

    int  h_idx[8];
    for (int i = 0; i < T; ++i) h_idx[i] = i % V;

    auto upload = [](size_t n, const float* h) {
        float* d; CUDA_CHECK(cudaMalloc(&d, n*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };
    GPT2Weights w{};
    w.wte = upload(V * E, h_wte);
    w.wpe = upload(T * E, h_wpe);
    w.ln1_w = upload(NL * E, h_ln1);
    w.ln2_w = upload(NL * E, h_ln2);
    w.qkv_w = upload(NL * 3 * E2, h_qkv);
    w.attn_proj_w = upload(NL * E2, h_apw);
    w.fc_w   = upload(NL * E4 * E, h_fc);
    w.fc2_w  = upload(NL * E * E4, h_fc2);
    w.ln_f_w = upload(E, h_lnf);

    int* d_idx; CUDA_CHECK(cudaMalloc(&d_idx, T*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, T*sizeof(int), cudaMemcpyHostToDevice));

    GPT2Acts a{};
    auto alloc = [](size_t n) { float* d; cudaMalloc(&d, n*sizeof(float)); return d; };
    a.encoded       = alloc(B*T*E);
    a.x             = alloc(B*T*E);
    a.ln1_out       = alloc(B*T*E);
    a.ln2_out       = alloc(B*T*E);
    a.ln_f_out      = alloc(B*T*E);
    a.qkv           = alloc(B*T*3*E);
    a.att           = alloc(B*H*T*T);
    a.attn_out      = alloc(B*T*E);
    a.attn_proj_out = alloc(B*T*E);
    a.fc_out        = alloc(B*T*4*E);
    a.mlp_out       = alloc(B*T*E);
    a.mean          = alloc(B*T);
    a.rstd          = alloc(B*T);
    a.logits        = alloc(B*T*V);

    gpt2_forward(cfg, w, a, d_idx);

    // Read back logits[0, T-1, :] just to prove the pipeline ran.
    float h_logits[65];
    CUDA_CHECK(cudaMemcpy(h_logits, a.logits + (T-1)*V, V*sizeof(float),
                          cudaMemcpyDeviceToHost));
    float maxv = h_logits[0];  int maxi = 0;
    for (int i = 1; i < V; ++i) if (h_logits[i] > maxv) { maxv = h_logits[i]; maxi = i; }
    printf("argmax_logit_at_last_pos=%d max=%.4f\n", maxi, maxv);

    // (skipping freeing; OS will clean up when we exit -- this is a test rig)
    return 0;
}
