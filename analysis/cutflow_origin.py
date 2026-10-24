#!/usr/bin/env python3
"""v2 analysis: full jet origin breakdown + cutflow (no cut / eta / pT / both) + plots.
Usage: python3 cutflow_origin.py [INDIR] [OUTDIR]
Reads {zj,ttZ,ttbb,bbbb}_2*_jets2.root (trees jets2, evts) from INDIR."""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.rcParams["pdf.fonttype"]=42
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

IN = sys.argv[1] if len(sys.argv) > 1 else "/eos/home-p/pkangazi/ZTAG_project/fastjet/out/v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(IN, "report")
os.makedirs(OUT, exist_ok=True)
PTCUT, ETACUT = 400.0, 2.0
MINEV = 1000      # jobs with fewer events (MadGraph shortfall) are skipped
SAMPLES = [("bbbb", r"$b\bar bb\bar b$ (background)"), ("ttbb", r"$t\bar tb\bar b$ (background)"),
           ("zj", r"$Z+$jet, $Z\to b\bar b$"), ("ttZ", r"$t\bar tZ$, $Z\to b\bar b$")]
CATS = [("Z->bb (both b in jet)", "tab:red"), ("Z partial (Z in cone, <2 b)", "tab:pink"),
        ("bb pair (>=2 b, not from top)", "tab:green"), ("single b (not from top)", "tab:olive"),
        ("b from top only", "tab:brown"), ("hard gluon", "tab:blue"),
        ("hard light quark", "tab:cyan"), ("other (ISR / UE / soft)", "tab:gray")]
CN = [c for c, _ in CATS]

def load(name):
    files = sorted(glob.glob(os.path.join(IN, f"{name}_[23][0-9][0-9]_jets2.root")))
    files = [f for f in files if not f.endswith("_399_jets2.root")]
    J, E, skipped = {}, {}, []
    for f in files:
        with uproot.open(f) as t:
            a = t["jets2"].arrays(library="np"); b = t["evts"].arrays(library="np")
        if len(b["evt"]) < MINEV:
            print(f"  skip {os.path.basename(f)}: only {len(b['evt'])} events"); skipped.append(f); continue
        for k, v in a.items(): J.setdefault(k, []).append(v)
        for k, v in b.items(): E.setdefault(k, []).append(v)
    if not files: return None, None, 0
    J = {k: np.concatenate(v) for k, v in J.items()}
    E = {k: np.concatenate(v) for k, v in E.items()}
    xs = []
    for f in sorted(glob.glob(os.path.join(IN, f"{name}_[23][0-9][0-9]_xsec.txt"))):
        if "_399_" in f: continue
        if any(os.path.basename(f).startswith(os.path.basename(x)[:-11]) for x in skipped): continue
        for line in open(f):
            if "Cross-section" in line: xs.append(float(line.split(":")[1].split("+-")[0]))
    return J, E, (np.mean(xs) if xs else float("nan"))

def category(J):
    n = len(J["pt"]); cat = np.full(n, 7, dtype=int)
    zb, zp = J["z_both"] > 0, J["z_part"] > 0
    nb, nt = J["nb_notop"], J["nb_top"]
    g, q = J["n_hard_g"] > 0, J["n_hard_q"] > 0
    done = np.zeros(n, bool)
    for i, m in enumerate([zb, zp & ~zb, (nb >= 2), (nb == 1), (nt >= 1) & (nb == 0), g, q]):
        m = m & ~done; cat[m] = i; done |= m
    return cat

data = {}
for s, _ in SAMPLES:
    J, E, xs = load(s)
    if J is None: print("no files for", s); continue
    J["cat"] = category(J); data[s] = (J, E, xs)

stages = [("no cut (pT>20)", lambda J: np.ones(len(J["pt"]), bool)),
          (f"abs(eta)<{ETACUT:g} only", lambda J: np.abs(J["eta"]) < ETACUT),
          (f"pT>{PTCUT:g} only", lambda J: J["pt"] > PTCUT),
          ("both cuts", lambda J: (np.abs(J["eta"]) < ETACUT) & (J["pt"] > PTCUT))]

md = ["# Jet cutflow with truth origin (anti-kT R=0.8)\n"]
csv = ["sample,events,n_all_jets_no_pt_cut,stage," + ",".join(CN) + ",total"]
counts = {}
for s, lab in SAMPLES:
    if s not in data: continue
    J, E, xs = data[s]
    nev = len(E["evt"]); nall = int(E["n_all_jets"].sum())
    md.append(f"\n## {s}: {nev} events, sigma={xs:.4g} pb, all clustered jets (no pT cut) = {nall} ({nall/nev:.1f}/event)\n")
    md.append("| stage | " + " | ".join(CN) + " | total | per event |")
    md.append("|---|" + "---|" * (len(CN) + 2))
    for sn, f in stages:
        m = f(J); c = np.bincount(J["cat"][m], minlength=len(CN)); counts[(s, sn)] = c
        md.append(f"| {sn} | " + " | ".join(str(int(x)) for x in c) + f" | {int(c.sum())} | {c.sum()/nev:.3f} |")
        csv.append(f"{s},{nev},{nall},{sn}," + ",".join(str(int(x)) for x in c) + f",{int(c.sum())}")
    # removed fractions
    c0, ce, cp, cb = (counts[(s, st[0])] for st in stages)
    md.append(f"\nremoved by eta (of pT>20 jets): {100*(1-ce.sum()/c0.sum()):.1f}%;  removed by pT>{PTCUT:g}: {100*(1-cp.sum()/c0.sum()):.1f}%;  left after both: {int(cb.sum())} ({cb.sum()/nev:.3f}/event)")
    md.append("\nfraction of jets surviving both cuts, per category: " + ", ".join(
        f"{CN[i]}: {100*cb[i]/max(c0[i],1):.1f}%" for i in range(len(CN)) if c0[i] > 0))
open(os.path.join(OUT, "cutflow.md"), "w").write("\n".join(md) + "\n")
open(os.path.join(OUT, "cutflow.csv"), "w").write("\n".join(csv) + "\n")
print("\n".join(md))

# ---------- plots ----------
pdf = PdfPages(os.path.join(OUT, "report_v2.pdf"))
def done(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

# (1) origin breakdown per stage: stacked bars
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
for ax, (s, lab) in zip(axs.ravel(), SAMPLES):
    if s not in data: ax.axis("off"); continue
    x = np.arange(len(stages)); bottom = np.zeros(len(stages))
    for i, (cn, col) in enumerate(CATS):
        h = np.array([counts[(s, sn)][i] for sn, _ in stages], float)
        ax.bar(x, h, bottom=bottom, color=col, label=cn); bottom += h
    ax.set_yscale("log"); ax.set_ylim(1, bottom.max() * 3)
    ax.set_xticks(x); ax.set_xticklabels([sn for sn, _ in stages], rotation=15, fontsize=8)
    ax.set_title(f"{lab}: jets by origin ({len(data[s][1]['evt'])} events)", fontsize=10)
    for xi, b in zip(x, bottom): ax.text(xi, b * 1.1, f"{int(b)}", ha="center", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
fig.subplots_adjust(bottom=0.14)
done(fig, "cutflow_stacked")

# (2) fractions of the origin per stage (normalised stacked bars)
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
for ax, (s, lab) in zip(axs.ravel(), SAMPLES):
    if s not in data: ax.axis("off"); continue
    x = np.arange(len(stages)); bottom = np.zeros(len(stages))
    tot = np.array([counts[(s, sn)].sum() for sn, _ in stages], float)
    for i, (cn, col) in enumerate(CATS):
        h = np.array([counts[(s, sn)][i] for sn, _ in stages], float) / tot
        ax.bar(x, h, bottom=bottom, color=col, label=cn); bottom += h
    ax.set_xticks(x); ax.set_xticklabels([sn for sn, _ in stages], rotation=15, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title(f"{lab}: origin fractions", fontsize=10)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
fig.subplots_adjust(bottom=0.14)
done(fig, "origin_fractions")

# (3) variable distributions per origin, jets after both cuts; pT and eta also at pT>20
VARS = [("pt", r"jet $p_T$ [GeV]", True), ("eta", r"jet $\eta$", False), ("mass", "jet mass [GeV]", False),
        ("nconst", "constituent multiplicity", False), ("girth", "girth", False)]
def panel(var, xl, logy, stage_name, f):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (s, lab) in zip(axs.ravel(), SAMPLES):
        if s not in data: ax.axis("off"); continue
        J = data[s][0]; m = f(J); x = J[var][m]
        if len(x) == 0: ax.axis("off"); continue
        lo, hi = np.percentile(x, [0.2, 99.8]); bins = np.linspace(lo, hi, 41)
        for i, (cn, col) in enumerate(CATS):
            sel = x[J["cat"][m] == i]
            if len(sel) < 100: continue
            h, _ = np.histogram(sel, bins); h = h / h.sum() / np.diff(bins)
            ax.step(bins, np.append(h, h[-1]), where="post", color=col, label=f"{cn} (N={len(sel)})")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(xl); ax.set_ylabel("normalised to unit area"); ax.set_title(f"{lab} - {stage_name}", fontsize=10)
        ax.legend(fontsize=6)
    return fig
for var, xl, logy in VARS:
    done(panel(var, xl, logy, "both cuts", stages[3][1]), f"{var}_bothcuts")
for var, xl, logy in VARS[:2]:
    done(panel(var, xl, logy, "no cut (pT>20)", stages[0][1]), f"{var}_nocut")

# (4) samples overlaid for each origin type with enough jets (after both cuts)
for i, (cn, col) in enumerate(CATS):
    ok = [s for s, _ in SAMPLES if s in data and ((data[s][0]["cat"][stages[3][1](data[s][0])] == i).sum() >= 100)]
    if len(ok) < 2: continue
    fig, axs = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (var, xl, logy) in zip(axs, VARS):
        for s in ok:
            J = data[s][0]; m = stages[3][1](J) & (J["cat"] == i); x = J[var][m]
            lo, hi = np.percentile(x, [0.2, 99.8]); bins = np.linspace(lo, hi, 36)
            h, _ = np.histogram(x, bins); h = h / h.sum() / np.diff(bins)
            ax.step(bins, np.append(h, h[-1]), where="post", label=f"{s} (N={len(x)})")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(xl); ax.legend(fontsize=7)
    fig.suptitle(f"origin: {cn} (after both cuts), by sample")
    done(fig, f"by_origin_{i}")
pdf.close()
print("report ->", OUT)


