"""Projected persistent KV storage per METHOD, llama-8b, from real run lengths.

Page-summary closed forms taken verbatim from analysis/plot_memory.py:
  min/max (FreeKV, ArkVale) : 2*d*B                      = 512 B /page/kvhead
  OVAL rank-8 bf16          : (d*r + page*r + d)*B       = 2816 B
  OVAL rank-8 int4/int8     : r*(d/2+2) + page*(r+2) + d+2 = 978 B
  RaaS                      : min/max but only budget/page pages (O(1) in ctx)

Crucial asymmetry: for FreeKV/ArkVale the summary sits ON TOP of the full key
cache. With --oval_recon the record IS the key store, so OVAL's summary
REPLACES the paged keys instead of adding to them.
"""
import glob, json
import numpy as np

L,NKV,d,pg,r,B = 32,8,128,32,8,2
sink=recent=512; budget=2048
scale=L*NKV; MiB=1/2**20
MM   = 2*d*B                                  # 512
OVL  = (d*r + pg*r + d)*B                     # 2816
OVLQ = r*(d//2+2) + pg*(r+2) + d+2            # 978
RES  = budget//pg                             # resident pages for RaaS
R="/home/shahbaa/oval-kv/results/codec/ds-r1-llama-8b-spec_ret"
NEXP={"MATH50":50,"AIME24":30,"GPQA50c":50}

print(f"llama-8b | {L}L x {NKV}kv, d={d}, page={pg}, rank={r} | per page per kv head: "
      f"min/max {MM}B, OVAL bf16 {OVL}B ({OVL/MM:.2f}x), OVAL int4 {OVLQ}B ({OVLQ/MM:.2f}x)\n")
hdr=(f"{'Task':<9}{'gens':>6}{'meanSeq':>9} | {'FullKV':>9}{'FreeKV/':>9}{'RaaS':>8} | "
     f"{'OVAL':>9}{'OVAL':>9} | {'bf16':>7}{'int4':>7}")
print(hdr)
print(f"{'':<9}{'':>6}{'':>9} | {'K+V':>9}{'ArkVale':>9}{'summ':>8} | "
      f"{'bf16':>9}{'int4':>9} | {'vs Full':>7}{'vs Full':>7}")
print("-"*len(hdr))
for task,n in NEXP.items():
    lens=[]
    for f in sorted(glob.glob(f"{R}/{task}-*oval_r{r}_eta*_q0_recon1*.jsonl")):
        rows=[json.loads(l) for l in open(f) if l.strip()]
        if len(rows)==n: lens+=[x["input_len"]+x["output_len"] for x in rows]
    if not lens: print(f"{task:<9} pending"); continue
    N=len(lens); dK=dV=mm=ov=ovq=retK=0.0
    for s in lens:
        npg=max(0,(s-sink-recent)//pg)
        mm  += npg*MM*scale
        ov  += npg*OVL*scale
        ovq += npg*OVLQ*scale
        retK+= (s-npg*pg)*d*B*scale
        dK  += s*d*B*scale
    dK/=N; dV=dK; mm/=N; ov/=N; ovq/=N; retK/=N
    full   = dK+dV
    freekv = dK+dV+mm            # summary ON TOP of full KV
    raas   = RES*MM*scale        # O(1), context-independent
    ovalb  = ov+retK+dV          # record REPLACES paged keys
    ovalq  = ovq+retK+dV
    print(f"{task:<9}{N:>6}{sum(lens)/N:>9.1f} | {full*MiB:>9.1f}{freekv*MiB:>9.1f}"
          f"{raas*MiB:>8.1f} | {ovalb*MiB:>9.1f}{ovalq*MiB:>9.1f} | "
          f"{ovalb/full:>7.3f}{ovalq/full:>7.3f}")
print(f"\nRaaS summary is {RES} resident pages = {RES*MM*scale*MiB:.1f} MiB at ANY context "
      f"(it drops pages; the others are O(context)).")
print("FreeKV/ArkVale column = full K + full V + min/max summary.")
print("OVAL columns = records + retained exact K + full V (records replace paged keys).")
