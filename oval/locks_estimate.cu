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
// Output layout matches estimate.cu exactly -- [bsz, n_qo_heads, n_pages] --
// so the caller applies the same reshape({bsz, n_groups, nqo/n_groups, L})
// .mean(2) group reduction and nothing downstream needs to change.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

#define LOCKS_MAX_RANK 16
#define LOCKS_MAX_PAGE 64

namespace {

template <typename T>
__device__ __forceinline__ float to_f(T x) { return static_cast<float>(x); }

// One block per (batch, page, kv_head). Threads cooperate over head_dim, then
// over the page's tokens. Query heads of the group are looped inside.
template <typename scalar_t, int BLOCK>
__global__ void locks_score_kernel(
    const scalar_t *__restrict__ q,       // [B, n_qo, d]
    const scalar_t *__restrict__ basis,   // [B, P, H, d, r]
    const scalar_t *__restrict__ coef,    // [B, P, H, T, r]
    const scalar_t *__restrict__ mu,      // [B, P, H, d]
    float *__restrict__ out,              // [B, n_qo, P]
    int P, int H, int d, int r, int T, int g, int n_qo) {
  const int p = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  const long pageBase = ((long)(b * P + p) * H + h);
  const scalar_t *B_ = basis + pageBase * d * r;
  const scalar_t *R_ = coef + pageBase * T * r;
  const scalar_t *M_ = mu + pageBase * d;

  __shared__ float sred[BLOCK];
  __shared__ float su[LOCKS_MAX_RANK];
  __shared__ float sc;
  __shared__ float stok[LOCKS_MAX_PAGE];

  for (int gi = 0; gi < g; ++gi) {
    const int qh = h * g + gi;
    const scalar_t *Q_ = q + ((long)b * n_qo + qh) * d;

    // u[rr] = sum_d basis[d][rr] * q[d]
    for (int rr = 0; rr < r; ++rr) {
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += to_f(B_[(long)dd * r + rr]) * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
      }
      if (tid == 0) su[rr] = sred[0];
      __syncthreads();
    }

    // c = mu . q
    {
      float acc = 0.f;
      for (int dd = tid; dd < d; dd += BLOCK)
        acc += to_f(M_[dd]) * to_f(Q_[dd]);
      sred[tid] = acc;
      __syncthreads();
      for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
      }
      if (tid == 0) sc = sred[0];
      __syncthreads();
    }

    // tok[t] = sum_r coef[t][rr]*u[rr] + c
    for (int t = tid; t < T; t += BLOCK) {
      float acc = sc;
      for (int rr = 0; rr < r; ++rr)
        acc += to_f(R_[(long)t * r + rr]) * su[rr];
      stok[t] = acc;
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

}  // namespace

// q [B, n_qo, d]; basis [B,P,H,d,r]; coef [B,P,H,T,r]; mu [B,P,H,d]
// returns [B, n_groups, P] -- their contract, group-reduced by mean.
torch::Tensor locks_estimate_scores(torch::Tensor q, torch::Tensor basis,
                                    torch::Tensor coef, torch::Tensor mu,
                                    int64_t n_groups) {
  TORCH_CHECK(q.is_cuda() && basis.is_cuda() && coef.is_cuda() && mu.is_cuda());
  TORCH_CHECK(basis.dim() == 5 && coef.dim() == 5 && mu.dim() == 4);
  const int B = basis.size(0), P = basis.size(1), H = basis.size(2);
  const int d = basis.size(3), r = basis.size(4), T = coef.size(3);
  const int n_qo = q.size(1);
  const int g = n_qo / H;
  TORCH_CHECK(r <= LOCKS_MAX_RANK, "rank ", r, " > ", LOCKS_MAX_RANK);
  TORCH_CHECK(T <= LOCKS_MAX_PAGE, "page_size ", T, " > ", LOCKS_MAX_PAGE);
  TORCH_CHECK(n_qo % H == 0);
  TORCH_CHECK(n_qo % n_groups == 0);

  auto out = torch::empty({B, n_qo, P},
                          q.options().dtype(torch::kFloat32));
  constexpr int BLOCK = 128;
  dim3 grid(P, H, B);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      q.scalar_type(), "locks_estimate_scores",
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        locks_score_kernel<at::Half, BLOCK><<<grid, BLOCK, 0, stream>>>(
            q.data_ptr<at::Half>(), basis.data_ptr<at::Half>(),
            coef.data_ptr<at::Half>(), mu.data_ptr<at::Half>(),
            out.data_ptr<float>(), P, H, d, r, T, g, n_qo);
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        locks_score_kernel<at::BFloat16, BLOCK><<<grid, BLOCK, 0, stream>>>(
            q.data_ptr<at::BFloat16>(), basis.data_ptr<at::BFloat16>(),
            coef.data_ptr<at::BFloat16>(), mu.data_ptr<at::BFloat16>(),
            out.data_ptr<float>(), P, H, d, r, T, g, n_qo);
      }));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "locks_score_kernel launch failed");

  // same group reduction as estimate.cu
  return out.reshape({B, (int)n_groups, n_qo / (int)n_groups, P})
      .mean(2)
      .to(q.scalar_type());
}
