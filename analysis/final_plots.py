import os, numpy as np
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import jetdata as sc
OUT = os.path.join(sc.IN, "finalplots")
NAME = {n: l for n, l, c in sc.SAMP}; COL = {n: c for n, l, c in sc.SAMP}

def hist(x, bins):
    h = np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9 * (bins[-1] - bins[0])), bins)[0].astype(float)
    n = max(len(x), 1); w = np.diff(bins)
    return h / n / w, np.sqrt(h) / n / w

def plot(curves, ratios, xl, title, fname, pdf, bins, logy=False, ratio_log=False, ylab="normalised to unit area"):
    c = 0.5 * (bins[1:] + bins[:-1])
    noratio = fname.startswith("s1_")
    if noratio:
        fig, a0 = plt.subplots(figsize=(7.5, 5.2)); a1 = fig.add_axes([0, 0, 0.01, 0.01]); a1.set_visible(False)
    else:
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.05))
    H = {}
    for k, x, N in curves:
        d, e = hist(x, bins); H[k] = (d, e)
        a0.step(bins, np.append(d, d[-1]), where="post", color=COL[k], lw=2, label=f"{NAME[k]}: {N}")
    a0.set_ylabel(ylab); a0.legend(frameon=False, fontsize=10); a0.set_title(title, fontsize=10)
    if logy: a0.set_yscale("log")
    for nk, dk, mk, filled in ([] if noratio else ratios):
        (dn, en), (dd, ed) = H[nk], H[dk]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = dn / dd; er = r * np.sqrt((en / dn) ** 2 + (ed / dd) ** 2)
        r[~np.isfinite(r) | (r <= 0)] = np.nan; er[~np.isfinite(er)] = 0
        a1.errorbar(c, r, er, fmt=mk, ms=4, color=COL[nk], mfc=COL[nk] if filled else "none", label=f"{NAME[nk]} / {NAME[dk]}")
    a1.axhline(1, color="gray", lw=1); a1.set_xlabel(xl); a1.set_xlim(bins[0], bins[-1]); a1.set_ylabel("ratio")
    if ratio_log: a1.set_yscale("log"); a1.set_ylim(0.05, 20)
    else: a1.set_ylim(0, 3)
    if noratio: a0.set_xlabel(xl); a0.set_xlim(bins[0], bins[-1])
    if len(ratios) > 1 and not noratio: a1.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, fname + ".png"), dpi=130, bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def selmask(J, k):
    fin = (J["pt"] > 400) & (np.abs(J["eta"]) < 2.0) & (J["nconst"] >= 3)
    sel = (J["z_both"] > 0) if k in ("zj", "ttZ") else ((J["nb_notop"] >= 2) & (J["z_both"] == 0) & (J["z_part"] == 0))
    return fin & sel

def nearest_dr(J, m):
    idx = np.flatnonzero(m); key = J["key"]
    lo = np.searchsorted(key, key[idx], "left"); hi = np.searchsorted(key, key[idx], "right"); out = []
    for i, a, b in zip(idx, lo, hi):
        if b - a < 2: continue
        sl = np.arange(a, b); sl = sl[sl != i]
        out.append(sc.dR(J["eta"][i], J["phi"][i], J["eta"][sl], J["phi"][sl]).min())
    return np.array(out)

def main(loader=sc.load_raw):
    os.makedirs(OUT, exist_ok=True)
    D = {}
    for n, l, c in sc.SAMP: D[n] = sc.build(loader(n))
    pdf1 = PdfPages(os.path.join(OUT, "set1_after_each_cut.pdf"))
    ST = [("pT20", "after pT>20 GeV"), ("eta", "after pT>20 and |eta|<2"), ("pt400", "after pT>20, |eta|<2 and pT>400 GeV"), ("sel", "after pT>400, |eta|<2, nconst>=3 and origin selection (final data)")]
    keys = [k for k, _, _ in sc.SAMP]
    ratios1 = [(k, "zj", "o", True) for k in keys[1:]]
    plot([(k, D[k][1].astype(float), int(D[k][1].sum())) for k in keys], ratios1, "all clustered jets per event", "no pT threshold in clustering", "s1_all_njets", pdf1, np.linspace(0, 80, 41), ylab="fraction of events")
    for si, (tag, ttl) in enumerate(ST):
        M = {k: (sc.stage_masks(D[k][2])[si][1] if si < 3 else selmask(D[k][2], k)) for k in keys}; Njets = {k: int(M[k].sum()) for k in keys}
        eta_r = (-5, 5) if si == 0 else (-2, 2); pt_r = (20, 1500) if si < 2 else (400, 1200)
        spec = [("njets", "jets per event", np.arange(0, 15), False),
                ("nconst", "constituents per jet", np.arange(0, 122, 2) - 0.5, False),
                ("dr", "$\\Delta R$(leading jet, 2nd jet)", np.linspace(0, 6, 31), False),
                ("pt", "jet $p_T$ [GeV]", np.linspace(*pt_r, 41), True),
                ("eta", "jet $\\eta$", np.linspace(*eta_r, 31), False),
                ("mass", "jet mass [GeV]", np.linspace(0, 300, 31), False),
                ("girth", "girth", np.linspace(0, 0.6, 31), False)]
        if si == 3: spec[2] = ("dr", "$\\Delta R$(selected jet, nearest other jet with pT>20)", np.linspace(0, 6, 31), False)
        for var, xl, bins, logy in spec:
            cur = []
            for k in keys:
                J = D[k][2]; m = M[k]
                x = (sc.per_event_counts(J, m, D[k][0]).astype(float) if var == "njets" else (nearest_dr(J, m) if si == 3 else sc.lead2_dr(J, m)) if var == "dr" else J[var][m].astype(float))
                cur.append((k, x, Njets[k]))
            plot(cur, ratios1, xl, ttl, f"s1_{tag}_{var}", pdf1, bins, logy=logy, ylab="fraction of events" if var == "njets" else "normalised to unit area")
    pdf1.close()
    S = {}
    for k in keys:
        J = D[k][2]; fin = (J["pt"] > 400) & (np.abs(J["eta"]) < 2.0) & (J["nconst"] >= 3)
        sel = (J["z_both"] > 0) if k in ("zj", "ttZ") else ((J["nb_notop"] >= 2) & (J["z_both"] == 0) & (J["z_part"] == 0))
        S[k] = {v: J[v][fin & sel] for v, _ in sc.SUBV}
    ttl = "after pT>400, |eta|<2, nconst>=3 and origin selection"
    sets = [("set2_ttZ_vs_Zj", ["zj", "ttZ"], [("ttZ", "zj", "o", True)], False),
            ("set3_ttbb_vs_bbbb", ["bbbb", "ttbb"], [("ttbb", "bbbb", "o", True)], False),
            ("set4_signal_vs_background", ["zj", "ttZ", "bbbb", "ttbb"],
             [("zj", "bbbb", "o", True), ("ttZ", "ttbb", "s", True)], True)]
    for name, ks, rat, rlog in sets:
        pdf = PdfPages(os.path.join(OUT, name + ".pdf"))
        for var, xl in sc.SUBV:
            allv = np.concatenate([S[k][var] for k in ks]); lo, hi = np.percentile(allv, [0.5, 99.5])
            bins = np.arange(int(lo), int(hi) + 2, 2) - 0.5 if var == "nconst" else np.linspace(lo, hi, 31)
            plot([(k, S[k][var], len(S[k][var])) for k in ks], rat, xl, ttl, f"{name}_{var}", pdf, bins, logy=(var == "pt"), ratio_log=False)
        pdf.close()
    print("done ->", OUT)

if __name__ == "__main__":
    main()
