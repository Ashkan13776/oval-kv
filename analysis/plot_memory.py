"""GPU/host memory for the KV-compression comparison.

Deliberately NOT a peak-memory bar chart. Every arm runs the same model at the
same budget and pred.py preallocates fixed arenas (6 GiB gpu / 20-40 GiB cpu)
independent of method, so peak allocation separates the arms by <5% and mostly
measures the harness. Left panel breaks out what each METHOD actually costs;
right panel shows how the page-summary cost scales with context, which is where
the designs genuinely diverge.

Reads results/freekv/memory/*.json, writes memory.png/.pdf beside them.
"""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM = os.path.join(ROOT, "results/freekv/memory")
ARMS = [("arkvale","ArkVale","#9e9e9e"), ("raas","RaaS\n(dropping)","#8c564b"),
        ("freekv","FreeKV\n(min/max)","#1f77b4"),
        ("oval0bf16","ours $\\eta$=0\nbf16","#98df8a"),
        ("oval0","ours $\\eta$=0\nint4","#2ca02c"),
        ("oval50bf16","ours $\\eta$=.5\nbf16","#ffbb78"),
        ("oval50","ours $\\eta$=.5\nint4","#ff7f0e"),
        ("oval100bf16","ours $\\eta$=1\nbf16","#ff9896"),
        ("oval100","ours $\\eta$=1\nint4","#d62728")]
# ONE page-summary row. Each method has exactly one structure representing
# pages -- min/max for FreeKV/ArkVale, a compact min/max for RaaS, a rank-8
# basis for ours -- so they belong in the same row. Splitting ours into
# "summary + method store" would imply an extra cost on top of a summary.
COMPONENTS = [("kv_used_MB","KV resident","#4c72b0"),
              ("digest_used_MB","page summary","#dd8452"),
              ("host_used_MB","host (offload)","#8172b3")]

def load():
    out={}
    for f in glob.glob(os.path.join(MEM,"*.json")):
        arm=os.path.basename(f)[:-5].split("_")[-1]
        d=json.load(open(f)); out[arm]=d.get("steady_state") or {}
    return out

def main():
    V=load()
    arms=[a for a in ARMS if a[0] in V]
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(15.5,5.0))

    x=np.arange(len(arms)); bottom=np.zeros(len(arms))
    for key,lbl,c in COMPONENTS:
        vals=np.array([V[a[0]].get(key,0.0) for a in arms])
        ax.bar(x,vals,bottom=bottom,label=lbl,color=c,edgecolor="black",linewidth=.5)
        bottom+=vals
    for xi,tot in zip(x,bottom):
        ax.text(xi,tot+40,f"{tot:.0f}",ha="center",fontsize=8)
    # our page summary, called out because it is the only row that differs
    for xi,a in zip(x,arms):
        d=V[a[0]].get("digest_used_MB",0.0)
        if d>100:
            ax.text(xi,V[a[0]].get("kv_used_MB",0)+d/2,f"{d:.0f}",ha="center",
                    va="center",fontsize=7.5,color="white",weight="bold")
    ax.set_ylim(0, bottom.max()*1.38)
    ax.set_xticks(x); ax.set_xticklabels([a[1] for a in arms],fontsize=7)
    ax.set_ylabel("MB attributable to the method")
    ax.set_title("What each method costs\nQwen-2.5-7B, 32K in, $\\mathcal{B}$=2048",fontsize=10)
    ax.legend(fontsize=8,loc="upper left"); ax.grid(axis="y",alpha=.25,lw=.5); ax.set_axisbelow(True)
    pk=[V[a[0]].get("torch_peak_alloc_MB",0) for a in arms]
    ax.text(.985,.985,f"peak alloc separates the arms\nby only {(max(pk)/min(pk)-1)*100:.1f}% "
            f"({min(pk):.0f}-{max(pk):.0f} MB)\n-- why it is not the headline",
            transform=ax.transAxes,ha="right",va="top",fontsize=7.5,color="#b00",
            bbox=dict(fc="white",ec="#b00",lw=.6,alpha=.95))

    # scaling: exact, from tensor shapes; validated against the measured points
    # int4/int8 records (LOCKS' r8i4 scheme), scales inline:
    #   basis r*(d/2+2) | coef page*(r+2) | mu d+2   = 978 B per page per kv head
    L,NKV,D,PAGE,RANK,B=28,4,128,32,8,2
    mm=2*NKV*D*B
    oval=NKV*(RANK*(D//2+2) + PAGE*(RANK+2) + D+2)
    ctx=np.array([8192,32768,131072,524288,1048576]); P=ctx//PAGE
    res=2048//PAGE
    ax2.plot(ctx,mm*L*P/(1<<20),"o-",color="#1f77b4",label="min/max (FreeKV, ArkVale)")
    ax2.plot(ctx,oval*L*P/(1<<20),"o-",color="#d62728",label="OVAL (ours), rank 8, int4")
    ax2.plot(ctx,[mm*L*res/(1<<20)]*len(ctx),"o-",color="#8c564b",label="RaaS (drops pages)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("context length (tokens)"); ax2.set_ylabel("page-summary memory on GPU (MB)")
    ax2.set_title("How it scales\nours costs 1.91$\\times$ min/max per page",fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=.25,lw=.5,which="both"); ax2.set_axisbelow(True)
    ax2.set_ylim(2, 3e4)
    _top=oval*L*(1048576//PAGE)/(1<<20)
    ax2.annotate(f"{_top/1024:.1f} GB of a 48 GB card",xy=(1048576,_top),xytext=(1.1e5,_top*2.2),
                 fontsize=7.5,color="#d62728",
                 arrowprops=dict(arrowstyle="->",color="#d62728",lw=.8))

    fig.suptitle("KV-compression memory — measured breakdown (left) and exact scaling (right); "
                 "RTX A6000 48GB",fontsize=11)
    fig.text(.5,.030,"Peak GPU memory is not the headline: the harness preallocates fixed 6 GiB gpu / "
             "20 GiB cpu arenas regardless of method, so it separates the arms by <5%.",
             ha="center",fontsize=7.8,color="#444")
    fig.text(.5,.008,"One page-summary row per method (min/max, compact min/max, rank-8 basis) counted as "
             "PAGES ACTUALLY USED. Our records are int4/int8 (LOCKS' r8i4 scheme): 978 B per page per kv head. "
             "RaaS is O(1) in context; the others are O(context).",
             ha="center",fontsize=7.8,color="#444")
    fig.tight_layout(rect=[0,.075,1,.93])
    for ext in ("png","pdf"):
        p=os.path.join(MEM,f"memory.{ext}"); fig.savefig(p,dpi=160); print("wrote",p)

if __name__=="__main__": main()
