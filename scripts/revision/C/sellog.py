"""Read-only page-selection logger for C5 (per-frame selection maps from the REAL runs).

The harness selects pages in kvc/patch/dynamic_attention.py with
`torch.topk(scores[b, H, P], page_budget, dim=-1)` (one call per layer >= skip_layer per decode
step; SnapKV: same call on its frozen scores). We swap that module's global `torch` for a proxy
whose `topk` forwards unchanged and, for exactly that call shape, adds the selected page ids to a
GPU counter (and the number of selectable pages to an "available" counter). Nothing the harness
computes changes; FreeKV's submodule is not edited.

Per question we then map pages to tokens (token = sink + page * page_size + [0, page_size)) and
tokens to frames, and store two 64-vectors in the answer line:
  sel_frame[f]   = sum over decisions of (#tokens of frame f inside a selected page)
  avail_frame[f] = sum over decisions of (#tokens of frame f inside a selectable page)
A decision is one (decode step, layer, KV head). sel / avail = fraction of frame f kept.
"""
import torch

MAXP = 4096


class _Proxy:
    def __init__(self, t):
        self._t = t
        self.on = False
        self.k = None
        self.sel = None
        self.avail = None

    def __getattr__(self, n):
        return getattr(self._t, n)

    def topk(self, x, k, *a, **kw):
        out = self._t.topk(x, k, *a, **kw)
        if self.on and k == self.k and x.dim() == 3 and kw.get("dim", a[0] if a else -1) == -1:
            idx = out.indices
            if self.sel is None or self.sel.device != idx.device:
                self.sel = self._t.zeros(MAXP, dtype=self._t.float64, device=idx.device)
                self.avail = self._t.zeros(MAXP, dtype=self._t.float64, device=idx.device)
            self.sel.index_add_(0, idx.reshape(-1), self._t.ones(idx.numel(), dtype=self._t.float64, device=idx.device))
            self.avail[: x.shape[-1]] += x.shape[0] * x.shape[1]
        return out


P = None


def install(page_budget):
    global P
    import kvc.patch.dynamic_attention as DA
    if not isinstance(DA.torch, _Proxy):
        DA.torch = _Proxy(DA.torch)
    P = DA.torch
    P.k = page_budget
    P.on = False
    return P


def start():
    if P is not None:
        P.on = True
        if P.sel is not None:
            P.sel.zero_(); P.avail.zero_()


def frames(sink, page_size, vis_lo, vis_hi, n_frames=64):
    """Stop logging; per-frame (sel, avail) token counts for the visual span [vis_lo, vis_hi)."""
    if P is None:
        return None
    P.on = False
    if P.sel is None or float(P.avail.sum()) == 0:
        return None
    import numpy as np
    sel, av = P.sel.cpu().numpy(), P.avail.cpu().numpy()
    npg = int((av > 0).nonzero()[0].max()) + 1
    tok = sink + np.arange(npg)[:, None] * page_size + np.arange(page_size)[None]   # [npg, page]
    per = (vis_hi - vis_lo) / n_frames
    fr = np.floor((tok - vis_lo) / per).astype(int)
    ok = (tok >= vis_lo) & (tok < vis_hi)
    sf, af = np.zeros(n_frames), np.zeros(n_frames)
    for p in range(npg):
        f = fr[p][ok[p]]
        if len(f):
            c = np.bincount(f, minlength=n_frames)[:n_frames]
            sf += sel[p] * c; af += av[p] * c
    return dict(sel_frame=sf.tolist(), avail_frame=af.tolist(), sel_pages=int(npg),
                decisions=float(av[0]) if av[0] > 0 else 0.0)
