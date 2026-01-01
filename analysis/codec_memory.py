"""Exact bytes of the OVAL key codec vs the raw key cache.

Closed form, so nothing is estimated: a page's record is basis (d x r) +
coefficients (page x r) + centroid (d). Verified against the slab allocation a
real run makes -- measured bytes/page match this to the byte at every rank.

Sink and recent tokens are never paged and stay exact in every method, so they
are counted uncompressed on both sides.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=28)      # ds-r1-qwen-7b
ap.add_argument("--kv_heads", type=int, default=4)
ap.add_argument("--head_dim", type=int, default=128)
ap.add_argument("--page", type=int, default=32)
ap.add_argument("--bytes_per_val", type=int, default=2)   # bfloat16
ap.add_argument("--ctx", type=int, default=16500)
ap.add_argument("--sink", type=int, default=512)
ap.add_argument("--recent", type=int, default=512)
ap.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
ap.add_argument("--packed", action="store_true",
                help="size the int4/int8 records instead of bfloat16")
A = ap.parse_args()

B, d, pg = A.bytes_per_val, A.head_dim, A.page
raw_page = pg * d * B
exact_tokens = A.sink + A.recent
npages = max(0, A.ctx - exact_tokens) // pg
scale = A.layers * A.kv_heads
MiB = 1 / 2**20

def record(r):
    """bfloat16 record, or the shipped int4/int8 packing (scales inline)."""
    if not A.packed:
        return (d * r + pg * r + d) * B
    return (d * r / 2 + r * 2) + (pg * r + pg * 2) + (d + 2)

k_raw = npages * raw_page * scale
v_raw = k_raw
exact = exact_tokens * d * B * scale * 2          # sink+recent, keys AND values

print(f"{A.layers} layers x {A.kv_heads} kv heads, d={d}, page={pg}, "
      f"{'int4/int8 packed' if A.packed else 'bfloat16'} records")
print(f"context {A.ctx} = {exact_tokens} exact (sink+recent) + "
      f"{A.ctx - exact_tokens} paged = {npages} pages\n")
print(f"{'rank':>5} {'eff':>4} {'record':>9} {'keys':>10} {'K ratio':>8} "
      f"{'K+V':>10} {'KV ratio':>9}")
print(f"{'':>5} {'':>4} {'B/pg/kvh':>9} {'MiB':>10} {'':>8} {'MiB':>10} {'':>9}")
base_kv = k_raw + v_raw + exact
print(f"{'raw':>5} {'-':>4} {raw_page:>9} {k_raw*MiB:>10.1f} {1.0:>7.2f}x "
      f"{base_kv*MiB:>10.1f} {1.0:>8.2f}x")
for r in A.ranks:
    eff = min(r, pg - 1, d)
    rec = record(eff)
    k_new = npages * rec * scale
    kv_new = k_new + v_raw + exact
    star = "*" if eff != r else " "
    print(f"{r:>5} {eff:>4}{star}{rec:>8.0f} {k_new*MiB:>10.1f} "
          f"{k_raw/k_new:>7.2f}x {kv_new*MiB:>10.1f} {base_kv/kv_new:>8.2f}x")

cross = (pg * d - d) / (d + pg)
print(f"\nBreak-even rank for the keys alone: {cross:.1f} (record == raw page);")
print("above it the bfloat16 codec EXPANDS the key store.")
print("* rank clamped to page_size-1 (centring costs one degree of freedom);")
print("  at that rank the reconstruction is exact.")
print("\nValues are never compressed by a KEY basis, which caps whole-cache")
print("compression well below the key-only ratio at every rank.")
