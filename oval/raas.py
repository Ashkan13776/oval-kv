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


def refresh_and_pick_victim(raas, layer_idx, scores, seq_len, alpha,
                            n_sink_pages, n_win_pages):
    """Update timestamps from page scores, then return the victim slot.

    scores : [bsz, n_groups, n_pages] from the engine's estimate_scores --
             already the min/max bound, group-reduced, exactly the quantity the
             reference thresholds.
    Returns an int slot index in the evictable middle range, or None.
    """
    ts = raas.ts[layer_idx]                       # [bsz, budget, n_groups]
    n = min(scores.shape[-1], ts.shape[1])
    # softmax over pages, as the reference does before thresholding
    w = torch.softmax(scores[..., :n].float(), dim=-1)   # [bsz, n_groups, n]
    hot = (w > alpha).permute(0, 2, 1)                    # [bsz, n, n_groups]
    ts[:, :n][hot] = int(seq_len)

    # victim: oldest timestamp among the evictable middle pages (sink and the
    # recent window are never candidates, matching the reference)
    lo, hi = n_sink_pages, max(n_sink_pages + 1, ts.shape[1] - n_win_pages)
    if hi <= lo:
        return None
    mid = ts[:, lo:hi].amin(dim=-1)               # min over groups -> [bsz, hi-lo]
    victim = int(mid.sum(dim=0).argmin().item()) + lo
    return victim
