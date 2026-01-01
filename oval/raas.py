"""RaaS (Retrieval-as-a-Stream) dropping policy for FreeKV's perf engine.

Fig. 6 of the FreeKV paper plots RaaS, but the released perf engine implements
only arkvale / cuda_cpy, so it has to be built here. Reference implementation is
the accuracy harness's `raas_attn` (accuracy/kvc/patch/dynamic_attention.py).

Why this is portable at all
---------------------------
At page_size == 1 RaaS needs full attention weights to refresh timestamps, which
the perf engine's fused FlashInfer decode never materialises -- porting it there
would mean adding an eager-attention path and the resulting bar would measure
that overhead rather than the method. But Fig. 6 uses page_size = 32, and at
page_size > 1 the reference uses a QUEST-STYLE min/max page score instead:

    q_min_k = q * min_k ; q_max_k = q * max_k
    page_weights = max(q_min_k, q_max_k).sum(-1)
    update_mask  = softmax(page_weights).mean(over group) > alpha

which is exactly the bound the engine's own `estimate_scores` already computes
over its min/max digest. So RaaS here is: their digest + their scorer, with
top-k+recall replaced by timestamp bookkeeping and permanent eviction.

Difference in page lifecycle (the reason for the kv_cache hook)
--------------------------------------------------------------
FreeKV never truly drops a page -- every page lives on CPU and selection recalls
it back, so its "eviction" just rotates which GPU slot holds the newest page.
RaaS drops permanently. We therefore choose the victim slot (oldest timestamp)
and skip the CPU backup, which is also what makes it fast: no offload, no recall.
"""
import torch


class RaasState:
    """Per-layer page timestamps, keyed by (batch, gpu page slot, group)."""

    def __init__(self, n_layers, bsz, budget, n_groups, device):
        self.ts = torch.zeros(n_layers, bsz, budget, n_groups,
                              dtype=torch.int32, device=device)
        self.budget = budget
        self.n_groups = n_groups

    def reset(self):
        self.ts.zero_()


def refresh_and_pick_victim(raas, layer_idx, scores, kvc, seq_len, alpha,
                            n_sink_pages, n_win_pages):
    """Update timestamps from page scores, then return the victim slot.

    scores : [bsz, n_groups, n_resident] from estimate_scores over the COMPACT
    digest (see RaasDigest), so column j already IS gpu slot j -- no gc2cc
    gather. That gather was needed only while we scored FreeKV's full digest,
    which is indexed by cpu page id and spans every page ever seen.

    The softmax must cover exactly the resident set: the reference holds
    page_budget pages, so alpha=0.02 sits just above the uniform weight
    1/64=0.0156 and is a live threshold. Softmaxing over ~1000 pages instead
    drives the mean to 0.001, alpha never fires, no timestamp is refreshed and
    eviction degenerates to a fixed slot.
    """
    ts = raas.ts[layer_idx]                       # [bsz, budget, n_groups]
    B, budget, G = ts.shape
    nres = min(scores.shape[-1], budget)
    if nres <= 0:
        return None
    w = torch.softmax(scores[..., :nres].float(), dim=-1)
    hot = (w > alpha).permute(0, 2, 1)                    # [bsz, nres, n_groups]
    ts[:, :nres][hot] = int(seq_len)

    # victim: oldest timestamp among evictable middle slots (sink and the recent
    # window are never candidates, matching the reference)
    lo, hi = n_sink_pages, max(n_sink_pages + 1, nres - n_win_pages)
    if hi <= lo:
        return None
    mid = ts[:, lo:hi].amin(dim=-1)               # min over groups
    return int(mid.sum(dim=0).argmin().item()) + lo


class RaasDigest:
    """Min/max page digest holding ONLY the pages RaaS actually keeps.

    Why this exists
    ---------------
    FreeKV's own digest spans every page ever seen -- it keeps them all on CPU
    and recalls, so its scorer is O(all pages). RaaS drops pages permanently,
    so a faithful RaaS has only `budget` pages in existence to score. Scoring
    the full digest instead is a porting artifact, and NOT a small one:
    microbenchmarking their estimate_scores kernel on Qwen-2.5-7B geometry
    (28 layers, 28 qo / 4 kv heads, d=128, page 32) gives

        64 digest pages   0.0179 ms/layer ->  0.50 ms/step
        1017 digest pages 0.1748 ms/layer ->  4.90 ms/step
        => +4.39 ms/step, 13.2% of the measured 33.36 ms RaaS bar.

    So the arm is laid out here as a paged cache with digest token j == GPU
    slot j, which lets THEIR estimate_scores kernel be reused unchanged while
    scoring exactly the resident set. Every arm then does the work its own
    method specifies, which is what makes the figure comparable.
    """

    def __init__(self, n_layers, bsz, budget, page_size, n_kv, d, dtype, device):
        self.page_size = page_size
        self.budget = budget
        self.bsz = bsz
        self.n_phys_b = (budget + page_size - 1) // page_size
        n_phys = bsz * self.n_phys_b
        self.buffer = torch.zeros(n_layers, n_phys, 2, page_size, n_kv, d,
                                  dtype=dtype, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.indices = torch.arange(n_phys, **i32).reshape(bsz, self.n_phys_b)
        self.indptr = torch.arange(0, n_phys + 1, self.n_phys_b, **i32)
        self._base = (torch.arange(bsz, device=device) * self.n_phys_b)
        self.n_tokens = [0] * n_layers
        self._device = device

    def last_page_lens(self, layer_idx):
        n = max(1, self.n_tokens[layer_idx])
        return torch.full((self.bsz,), (n - 1) % self.page_size + 1,
                          dtype=torch.int32, device=self._device)

    def write_slot(self, layer_idx, slot, maxs, mins):
        """maxs / mins: [bsz, n_kv, d] digest of the page now living in `slot`."""
        p = slot // self.page_size
        off = slot % self.page_size
        idx = self._base + p
        buf = self.buffer[layer_idx]
        buf[idx, 0, off] = maxs.to(buf.dtype)
        buf[idx, 1, off] = mins.to(buf.dtype)
        if slot + 1 > self.n_tokens[layer_idx]:
            self.n_tokens[layer_idx] = slot + 1

    def write_all(self, layer_idx, maxs, mins):
        """Bulk-populate slots 0..n-1 after prefill. maxs/mins: [bsz,n,n_kv,d]."""
        n = maxs.shape[1]
        j = torch.arange(n, device=maxs.device)
        p = (j // self.page_size).to(torch.long)
        off = (j % self.page_size).to(torch.long)
        buf = self.buffer[layer_idx].view(self.bsz, self.n_phys_b, 2,
                                          self.page_size, *self.buffer.shape[-2:])
        # NB: buf[:, p, 0, off] would separate the two advanced indices with an
        # integer, which pushes the advanced dims to the front and yields
        # [n, bsz, ...]. Slice the k/v dim first so p and off stay adjacent.
        buf[:, :, 0][:, p, off] = maxs.to(buf.dtype)
        buf[:, :, 1][:, p, off] = mins.to(buf.dtype)
        self.n_tokens[layer_idx] = n
