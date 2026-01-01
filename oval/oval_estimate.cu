// Rank-r output-aware page scorer, the CUDA counterpart of their EstimateScores.
//
// Their digest is a (max,min) pair per page and the score is the Quest bound
//     sum_d max(q_d*min_d, q_d*max_d).
// Ours is a rank-r basis of the page's keys and the score is the page's log
// attention mass under the rank-r reconstruction:
//     u   = B^T q                       [r]
//     tok = R u + (mu . q)              [page]
//     s   = logsumexp_t tok_t
//
// STORAGE IS QUANTIZED, matching LOCKS' own r8i4 records (r8i4_build.py):
//   basis  int4, per-COLUMN bf16 scale, packed two nibbles per byte,
//          LO nibble = EVEN head_dim row
//   coef   int8, per-TOKEN bf16 scale
//   mu     int8, per-PAGE  bf16 scale
// Each row's scale lives INLINE in that row's trailing 2 bytes, so the tensor
// count and every signature are unchanged, and the scale shares a cache line
// with the data it scales. This cuts the bytes read per page from 11264 to
// 3912 (2.88x); the scorer is bandwidth-bound in exactly those bytes.
//
// Output layout matches estimate.cu exactly -- [bsz, n_qo_heads, n_pages] --
// so the caller applies the same reshape({bsz, n_groups, nqo/n_groups, L})
// .mean(2) group reduction and nothing downstream needs to change.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

#define OVAL_MAX_RANK 16
#define OVAL_MAX_PAGE 64

namespace {

template <typename T>
__device__ __forceinline__ float to_f(T x) { return static_cast<float>(x); }

// two little-endian bytes holding a bf16 -> float (exact: bf16 is the high
// half of an fp32)
__device__ __forceinline__ float bf16_at(const unsigned char *p) {
  unsigned int raw = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
  unsigned int bits = raw << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

// signed 4-bit field of a packed byte pair; lo nibble is the EVEN row
__device__ __forceinline__ int nib(unsigned char b, int odd) {
  int n = odd ? (b >> 4) : (b & 0xF);
  return n >= 8 ? n - 16 : n;
}

// One block per (batch, page, kv_head). Threads cooperate over head_dim, then
// over the page's tokens. Query heads of the group are looped inside.
template <typename scalar_t, int BLOCK>
__global__ void oval_score_kernel(
    const scalar_t *__restrict__ q,          // [B, n_qo, d]
    const unsigned char *__restrict__ vrow,  // [B, P, H, r, d/2+2]
    const signed char *__restrict__ crow,    // [B, P, H, T, r+2]
    const signed char *__restrict__ mrow,    // [B, P, H, d+2]
    float *__restrict__ out,                 // [B, n_qo, P]
    int P, int H, int d, int r, int T, int g, int n_qo) {
  const int p = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  const int vw = d / 2 + 2, cw = r + 2, mw = d + 2;
  const long pageBase = ((long)(b * P + p) * H + h);
  const unsigned char *V_ = vrow + pageBase * (long)r * vw;
  const signed char *C_ = crow + pageBase * (long)T * cw;
  const signed char *M_ = mrow + pageBase * (long)mw;

  __shared__ float sred[BLOCK];
  __shared__ float su[OVAL_MAX_RANK];
  __shared__ float sc;
  __shared__ float stok[OVAL_MAX_PAGE];

  for (int gi = 0; gi < g; ++gi) {
    const int qh = h * g + gi;
    const scalar_t *Q_ = q + ((long)b * n_qo + qh) * d;

    // u[rr] = vs[rr] * sum_d vq[rr][d] * q[d]
    for (int rr = 0; rr < r; ++rr) {
      const unsigned char *vr = V_ + (long)rr * vw;
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += (float)nib(vr[dd >> 1], dd & 1) * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
      }
      if (tid == 0) su[rr] = sred[0] * bf16_at(vr + d / 2);
      __syncthreads();
    }

    // c = mus * sum_d mu8[d] * q[d]
    {
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += (float)M_[dd] * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
      }
      if (tid == 0)
        sc = sred[0] * bf16_at((const unsigned char *)M_ + d);
      __syncthreads();
    }

    // tok[t] = cs[t] * sum_r c8[t][rr]*u[rr] + c
    for (int t = tid; t < T; t += BLOCK) {
      const signed char *cr = C_ + (long)t * cw;
      float acc = 0.f;
      for (int rr = 0; rr < r; ++rr) acc += (float)cr[rr] * su[rr];
      stok[t] = acc * bf16_at((const unsigned char *)cr + r) + sc;
    }
    __syncthreads();

    // logsumexp over the page's tokens (single thread; T <= 64)
    if (tid == 0) {
      float m = -INFINITY;
      for (int t = 0; t < T; ++t) m = fmaxf(m, stok[t]);
      float sum = 0.f;
      for (int t = 0; t < T; ++t) sum += __expf(stok[t] - m);
      out[((long)b * n_qo + qh) * P + p] = m + __logf(sum);
    }
    __syncthreads();
  }
}

// Unquantized variant, kept so bf16 and int4 records can be measured
// head-to-head in one build. Selected by the dtype of `basis`.
template <typename scalar_t, int BLOCK>
__global__ void oval_score_kernel_bf16(
    const scalar_t *__restrict__ q,       // [B, n_qo, d]
    const scalar_t *__restrict__ basis,   // [B, P, H, d, r]
    const scalar_t *__restrict__ coef,    // [B, P, H, T, r]
    const scalar_t *__restrict__ mu,      // [B, P, H, d]
    float *__restrict__ out,              // [B, n_qo, P]
    int P, int H, int d, int r, int T, int g, int n_qo) {
  const int p = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x;
  const long pageBase = ((long)(b * P + p) * H + h);
  const scalar_t *B_ = basis + pageBase * d * r;
  const scalar_t *R_ = coef + pageBase * T * r;
  const scalar_t *M_ = mu + pageBase * d;
  __shared__ float sred[BLOCK];
  __shared__ float su[OVAL_MAX_RANK];
  __shared__ float sc;
  __shared__ float stok[OVAL_MAX_PAGE];
  for (int gi = 0; gi < g; ++gi) {
    const int qh = h * g + gi;
    const scalar_t *Q_ = q + ((long)b * n_qo + qh) * d;
    for (int rr = 0; rr < r; ++rr) {
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += to_f(B_[(long)dd * r + rr]) * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s2 = BLOCK / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) sred[tid] += sred[tid + s2];
        __syncthreads();
      }
      if (tid == 0) su[rr] = sred[0];
      __syncthreads();
    }
    {
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += to_f(M_[dd]) * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s2 = BLOCK / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) sred[tid] += sred[tid + s2];
        __syncthreads();
      }
      if (tid == 0) sc = sred[0];
      __syncthreads();
    }
    for (int t = tid; t < T; t += BLOCK) {
      float acc = sc;
      for (int rr = 0; rr < r; ++rr)
        acc += to_f(R_[(long)t * r + rr]) * su[rr];
      stok[t] = acc;
    }
    __syncthreads();
    if (tid == 0) {
      float m = -INFINITY;
      for (int t = 0; t < T; ++t) m = fmaxf(m, stok[t]);
      float sum = 0.f;
      for (int t = 0; t < T; ++t) sum += __expf(stok[t] - m);
      out[((long)b * n_qo + qh) * P + p] = m + __logf(sum);
    }
    __syncthreads();
  }
}

}  // namespace

// q [B,n_qo,d]; vrow [B,P,H,r,d/2+2] u8; crow [B,P,H,T,r+2] i8; mrow [B,P,H,d+2] i8
// returns [B, n_groups, P] -- their contract, group-reduced by mean.
torch::Tensor oval_estimate_scores(torch::Tensor q, torch::Tensor basis,
                                    torch::Tensor coef, torch::Tensor mu,
                                    int64_t n_groups) {
  TORCH_CHECK(q.is_cuda() && basis.is_cuda() && coef.is_cuda() && mu.is_cuda());
  TORCH_CHECK(basis.dim() == 5 && coef.dim() == 5 && mu.dim() == 4);
  const bool quant = basis.scalar_type() == at::ScalarType::Byte;
  const int B = basis.size(0), P = basis.size(1), H = basis.size(2);
  int r, d;
  if (quant) {
    TORCH_CHECK(coef.scalar_type() == at::ScalarType::Char, "coef must be int8");
    TORCH_CHECK(mu.scalar_type() == at::ScalarType::Char, "mu must be int8");
    r = basis.size(3);
    d = (basis.size(4) - 2) * 2;
    TORCH_CHECK(coef.size(4) == r + 2, "coef row width mismatch");
    TORCH_CHECK(mu.size(3) == d + 2, "mu row width mismatch");
  } else {
    d = basis.size(3);
    r = basis.size(4);
  }
  const int T = coef.size(3);
  const int n_qo = q.size(1);
  const int g = n_qo / H;
  TORCH_CHECK(r <= OVAL_MAX_RANK, "rank ", r, " > ", OVAL_MAX_RANK);
  TORCH_CHECK(T <= OVAL_MAX_PAGE, "page_size ", T, " > ", OVAL_MAX_PAGE);
  TORCH_CHECK(n_qo % H == 0);
  TORCH_CHECK(n_qo % n_groups == 0);

  auto out = torch::empty({B, n_qo, P}, q.options().dtype(torch::kFloat32));
  constexpr int BLOCK = 128;
  dim3 grid(P, H, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (quant) {
    const unsigned char *vp = basis.data_ptr<unsigned char>();
    const signed char *cp = coef.data_ptr<signed char>();
    const signed char *mp = mu.data_ptr<signed char>();
    AT_DISPATCH_SWITCH(
        q.scalar_type(), "oval_estimate_scores",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
          oval_score_kernel<at::Half, BLOCK><<<grid, BLOCK, 0, stream>>>(
              q.data_ptr<at::Half>(), vp, cp, mp, out.data_ptr<float>(),
              P, H, d, r, T, g, n_qo);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
          oval_score_kernel<at::BFloat16, BLOCK><<<grid, BLOCK, 0, stream>>>(
              q.data_ptr<at::BFloat16>(), vp, cp, mp, out.data_ptr<float>(),
              P, H, d, r, T, g, n_qo);
        }));
  } else {
    AT_DISPATCH_SWITCH(
        q.scalar_type(), "oval_estimate_scores_bf16",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
          oval_score_kernel_bf16<at::Half, BLOCK><<<grid, BLOCK, 0, stream>>>(
              q.data_ptr<at::Half>(), basis.data_ptr<at::Half>(),
              coef.data_ptr<at::Half>(), mu.data_ptr<at::Half>(),
              out.data_ptr<float>(), P, H, d, r, T, g, n_qo);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
          oval_score_kernel_bf16<at::BFloat16, BLOCK><<<grid, BLOCK, 0, stream>>>(
              q.data_ptr<at::BFloat16>(), basis.data_ptr<at::BFloat16>(),
              coef.data_ptr<at::BFloat16>(), mu.data_ptr<at::BFloat16>(),
              out.data_ptr<float>(), P, H, d, r, T, g, n_qo);
        }));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "oval_score_kernel launch failed");

  // same group reduction as estimate.cu
  return out.reshape({B, (int)n_groups, n_qo / (int)n_groups, P})
      .mean(2)
      .to(q.scalar_type());
}
