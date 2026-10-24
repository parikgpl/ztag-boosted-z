import glob, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
IN = "/eos/home-p/pkangazi/ZTAG_project/fastjet/out/v5"
OUT = os.path.join(IN, "stages")
SAMP = [("zj", "Z+jet", "tab:purple"), ("ttZ", r"$t\bar tZ$", "tab:orange"),
        ("bbbb", r"$b\bar bb\bar b$", "tab:green"), ("ttbb", r"$t\bar tb\bar b$", "tab:brown")]
SUBV = [("pt", r"jet $p_T$ [GeV]"), ("eta", r"jet $\eta$"), ("mass", "jet mass [GeV]"), ("nconst", "constituent multiplicity"),
        ("girth", "girth"), ("tau1", r"$\tau_1$"), ("tau2", r"$\tau_2$"), ("tau3", r"$\tau_3$"), ("tau21", r"$\tau_{21}$"),
        ("tau32", r"$\tau_{32}$"), ("c2", r"$C_2$"), ("d2", r"$D_2$"), ("n2", r"$N_2$"), ("msd", "soft-drop mass [GeV]"),
        ("zg", r"$z_g$"), ("rg", r"$R_g$")]

def load_raw(name):
    import uproot
    L = []
    for f in sorted(glob.glob(f"{IN}/{name}_3[0-9][0-9]_jets2.root")):
        if "_399_" in f: continue
        with uproot.open(f) as t:
            e = t["evts"].arrays(library="np")
            if len(e["evt"]) < 1000: continue
            a = t["jets2"].arrays(library="np")
        L.append((len(e["evt"]), e["n_all_jets"], a))
    return L

def build(L, off=10**6):
    nev = 0; nall = []; J = {}
    for fi, (n, na, a) in enumerate(L):
        nev += n; nall.append(na)
        for k, v in a.items(): J.setdefault(k, []).append(v)
        J.setdefault("key", []).append(a["evt"].astype(np.int64) + fi * off)
    J = {k: np.concatenate(v) for k, v in J.items()}
    return nev, np.concatenate(nall), J

def dR(e1, p1, e2, p2):
    dp = (p1 - p2 + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(e1 - e2, dp)

def stage_masks(J):
    s1 = J["pt"] > 20; s2 = s1 & (np.abs(J["eta"]) < 2.0); s3 = s2 & (J["pt"] > 400)
    return [("pT>20", s1), ("+|eta|<2", s2), ("+pT>400 (final cuts)", s3)]

def per_event_counts(J, m, nev):
    k = J["key"][m]; u, c = np.unique(k, return_counts=True)
    return np.concatenate([c, np.zeros(nev - len(u), int)])

def lead2_dr(J, m):
    idx = np.flatnonzero(m); k = J["key"][idx]
    u, first, cnt = np.unique(k, return_index=True, return_counts=True)
    ok = cnt >= 2; a = idx[first[ok]]; b = idx[first[ok] + 1]
    return dR(J["eta"][a], J["phi"][a], J["eta"][b], J["phi"][b])

def ratio_plot(curves, xl, fname, pdf, bins, logy=False, integer=False, ylab="normalised to unit area"):
    c = 0.5 * (bins[1:] + bins[:-1]); w = np.diff(bins)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
    H = []
    for lab, col, x in curves:
        h, _ = np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9 * (bins[-1] - bins[0])), bins); n = max(len(x), 1)
        d = h / n / w; e = np.sqrt(h) / n / w; H.append((d, e))
        a0.step(bins, np.append(d, d[-1]), where="post", color=col, lw=2, label=f"{lab} (N={len(x)})")
    a0.set_ylabel(ylab); a0.legend(frameon=False, fontsize=9)
    if logy: a0.set_yscale("log")
    d0, e0 = H[0]
    for (lab, col, x), (d, e) in list(zip(curves, H))[1:]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = d / d0; er = r * np.sqrt((e / d) ** 2 + (e0 / d0) ** 2)
        r[~np.isfinite(r)] = np.nan; er[~np.isfinite(er)] = 0
        a1.errorbar(c, r, er, fmt="o", ms=3, color=col, label=lab)
    a1.axhline(1, color="gray", lw=1); a1.set_ylim(0, 3); a1.set_ylabel(f"/ {curves[0][0]}"); a1.set_xlabel(xl); a1.set_xlim(bins[0], bins[-1])
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=130, bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def main(loader=load_raw):
    os.makedirs(OUT, exist_ok=True)
    D = {}
    for name, lab, col in SAMP:
        nev, nall, J = build(loader(name)); D[name] = (nev, nall, J)
        print(name, "events", nev, "mean all-clustered jets/event %.1f" % nall.mean())
    pdfA = PdfPages(os.path.join(OUT, "stages_v3.pdf"))
    ratio_plot([(lab, col, D[n][1]) for n, lab, col in SAMP], "all clustered jets per event (no pT cut)", "njets_allclustered", pdfA,
               np.linspace(0, 80, 41), ylab="fraction of events")
    for si, sname in enumerate(["pT>20", "pT>20 & |eta|<2", "pT>20 & |eta|<2 & pT>400"]):
        cn = []; cc = []; cd = []
        for n, lab, col in SAMP:
            nev, _, J = D[n]; m = stage_masks(J)[si][1]
            cn.append((lab, col, per_event_counts(J, m, nev).astype(float)))
            cc.append((lab, col, J["nconst"][m].astype(float)))
            cd.append((lab, col, lead2_dr(J, m)))
        mx = max(8, int(max(c[2].max() for c in cn)) + 1)
        ratio_plot(cn, f"jets per event [{sname}]", f"njets_stage{si+1}", pdfA, np.arange(0, min(mx, 14) + 1), ylab="fraction of events")
        ratio_plot(cc, f"constituents per jet [{sname}]", f"nconst_stage{si+1}", pdfA, np.linspace(0, 120, 41))
        ratio_plot(cd, rf"$\Delta R$(leading jet, 2nd jet) [{sname}]", f"dR_stage{si+1}", pdfA, np.linspace(0, 6, 31))
    pdfA.close()
    pdfB = PdfPages(os.path.join(OUT, "sig_vs_bkg_v3.pdf"))
    S = {}
    for n, lab, col in SAMP:
        J = D[n][2]; fin = (J["pt"] > 400) & (np.abs(J["eta"]) < 2.0) & (J["nconst"] >= 3)
        sel = (J["z_both"] > 0) if n in ("zj", "ttZ") else ((J["nb_notop"] >= 2) & (J["z_both"] == 0) & (J["z_part"] == 0))
        S[n] = {k: J[k][fin & sel] for k in [v for v, _ in SUBV]}
    for var, xl in SUBV:
        allv = np.concatenate([S[n][var] for n, _, _ in SAMP]); lo, hi = np.percentile(allv, [0.5, 99.5])
        ratio_plot([(lab, col, S[n][var]) for n, lab, col in SAMP], xl, f"sb_{var}", pdfB, (np.arange(int(lo), int(hi) + 2, 2) - 0.5 if var == "nconst" else np.linspace(lo, hi, 31)), logy=(var == "pt"))
    pdfB.close(); print("done ->", OUT)

if __name__ == "__main__":
    main()
