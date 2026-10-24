"""plot_cons.py : the 17 features of the 10 hardest constituents of every final jet, from h5/v5.
usage: python3 plot_cons.py [all|full]     all = full + partial origin (default), full = origin 1 only
output: h5/v5/cons_plots_<mode>.pdf, one page per feature; 10 columns = particle rank 1..10;
top row: 4 samples (normalised), bottom row: ratios Z+jet/bbbb and ttZ/ttbb"""
import sys, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42
import os
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = "/eos/home-p/pkangazi/ZTAG_project/h5/v5"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
NR = 10
SAMP = [("Z+jet", "tab:purple"), ("ttZ", "tab:orange"), ("bbbb", "tab:green"), ("ttbb", "tab:brown")]   # sample id 0..3
PAIRS = [(0, 2, "o", "tab:purple"), (1, 3, "s", "tab:orange")]                                        # signal / background
PDG = [11, 13, 22, 130, 211, 321, 2212]
LOG = {"pt", "e", "erel", "ptrel"}
NAME = {"px": "$p_x$", "py": "$p_y$", "pz": "$p_z$", "e": "$E$", "erel": "$E/E_{jet}$", "pt": "$p_T$", "ptrel": "$p_T/p_{T,jet}$",
        "eta": r"$\eta$", "etarel": r"$\eta-\eta_{jet}$", "etarot": r"$\eta$ (rotated)", "phi": r"$\phi$", "phirel": r"$\phi-\phi_{jet}$",
        "phirot": r"$\phi$ (rotated)", "deltaR": r"$\Delta R$ (to jet axis)", "costheta": r"$\cos\theta$",
        "costhetarel": r"$\cos\theta_{rel}$ (to jet)", "pdgid": "|pdgId|"}

with h5py.File(f"{D}/jet_features.h5", "r") as h: sample = h["sample"][:]; origin = h["origin"][:]
with h5py.File(f"{D}/constituents.h5", "r") as h:
    feats = [x.decode() if isinstance(x, bytes) else x for x in h["constituents"].attrs["features"]]
    C = h["constituents"][:, :NR, :]                      # (N, 10, 17): only the 10 hardest
sel = np.ones(len(sample), bool) if mode == "all" else (origin == 1)
sample, C = sample[sel], C[sel]
ipt = feats.index("pt")
print("mode", mode, "| jets", len(sample), {s: int((sample == i).sum()) for i, (s, _) in enumerate(SAMP)})

pdf = PdfPages(f"{D}/cons_plots_{mode}.pdf")
PDFDIR = f"{D}/cons_pdf_{mode}"; os.makedirs(PDFDIR, exist_ok=True)
for fi, f in enumerate(feats):
    fig = plt.figure(figsize=(16, 11)); outer = GridSpec(2, 1, figure=fig, hspace=0.30)
    for r in range(NR):
        if r % 5 == 0: inner = outer[r // 5].subgridspec(2, 5, height_ratios=[3, 1.2], hspace=0.07, wspace=0.32)
        valid = C[:, r, ipt] > 0                          # particle exists (zero padded otherwise)
        v = C[:, r, fi]
        cc = r % 5; a0 = fig.add_subplot(inner[0, cc]); a1 = fig.add_subplot(inner[1, cc], sharex=a0)
        H = {}
        if f == "pdgid":
            for i in range(4):
                x = np.abs(v[valid & (sample == i)]).astype(int)
                H[i] = np.array([(x == p).sum() for p in PDG], float) / max(len(x), 1)
            cen = np.arange(len(PDG)); edges = np.arange(len(PDG) + 1) - 0.5
        else:
            pool = v[valid]
            lo, hi = np.percentile(pool, [0.5, 99.5]) if len(pool) else (0., 1.)
            if hi <= lo: lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, 31); cen = 0.5 * (edges[1:] + edges[:-1])
            for i in range(4):
                x = np.clip(v[valid & (sample == i)], edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0]))
                h, _ = np.histogram(x, edges)
                H[i] = h / max(h.sum(), 1) / (edges[1] - edges[0])
        for i, (lab, col) in enumerate(SAMP):
            a0.step(edges, np.append(H[i], H[i][-1]), where="post", color=col, lw=1.5, label=lab)
        if f in LOG: a0.set_yscale("log")
        else: a0.set_ylim(bottom=0)
        a0.set_title(f"particle {r + 1}", fontsize=10); a0.tick_params(labelbottom=False)
        a0.tick_params(labelsize=8)
        if cc == 0: a0.set_ylabel("normalised to unit area", fontsize=9); a0.legend(fontsize=7, frameon=False)
        for num, den, mk, col in PAIRS:
            with np.errstate(divide="ignore", invalid="ignore"): a1.plot(cen, H[num] / H[den], mk, ms=3, color=col)
        a1.axhline(1, color="gray", lw=0.8); a1.set_ylim(0, 3); a1.set_xlim(edges[0], edges[-1])
        a1.tick_params(labelsize=8); a1.set_yticks([0, 1, 2, 3])
        if f != "pdgid": a1.xaxis.set_major_locator(MaxNLocator(4)); a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
        if cc == 0: a1.set_ylabel("sig/bkg", fontsize=9)
        if f == "pdgid": a1.set_xticks(cen); a1.set_xticklabels([str(p) for p in PDG], rotation=90, fontsize=8)
        a1.set_xlabel(NAME.get(f, f) if f != "pdgid" else "|pdgId|", fontsize=9)
    fig.suptitle(f"{NAME.get(f, f)}: 10 hardest constituents of the final jets ({'full + partial' if mode == 'all' else 'full only'}); ratios: Z+jet/bbbb (purple), ttZ/ttbb (orange)", y=0.96, fontsize=12)
    fig.savefig(f"{PDFDIR}/cons_{mode}_{fi + 1:02d}_{f}.pdf", bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
pdf.close(); print("written", f"{D}/cons_plots_{mode}.pdf")
