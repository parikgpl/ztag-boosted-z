"""Jet-level numbers quoted in the report: medians of the full-origin Z jets in zj and ttZ (with and
without b from top) from h5/v5, and the distribution of the number of selected jets per event."""
import glob, numpy as np, h5py, uproot
B = "/eos/user/p/pkangazi/ZTAG_project/"
j = h5py.File(B + "h5/v5/jet_features.h5")
d = {k: j[k][:] for k in ["sample", "origin", "eta", "mass", "nconst", "girth", "tau21", "d2", "nb_top"]}
full = d["origin"] == 1
for s, n in ((0, "zj"), (1, "ttZ")):
    m = (d["sample"] == s) & full
    print(n, "N=%d" % m.sum(), " ".join("%s med=%.3f mean=%.3f" % (k, np.median(d[k][m]), d[k][m].mean()) for k in ["mass", "nconst", "girth", "tau21", "d2"]), "| frac nb_top>0=%.3f" % ((d["nb_top"][m] > 0).mean()))
    m2 = m & (d["nb_top"] == 0)
    print(n, "no b-from-top N=%d" % m2.sum(), " ".join("%s med=%.3f" % (k, np.median(d[k][m2])) for k in ["mass", "nconst", "girth", "tau21", "d2"]))
IN = B + "fastjet/out/v5/"
for s in ("zj", "ttZ", "bbbb", "ttbb"):
    cnt = []
    for f in sorted(glob.glob(IN + "%s_3[0-9][0-9]_jets2.root" % s)):
        with uproot.open(f) as t:
            e = t["evts"].num_entries
            if e < 1000: continue
            J = t["jets2"].arrays(["evt", "pt", "eta", "z_both", "z_part", "nb_notop"], library="np")
        c = (J["pt"] > 400) & (abs(J["eta"]) < 2)
        sel = c & (J["z_both"] > 0) if s in ("zj", "ttZ") else c & (J["nb_notop"] >= 2) & (J["z_both"] == 0) & (J["z_part"] == 0)
        cnt.append(np.bincount(J["evt"][sel], minlength=e))
    c = np.concatenate(cnt)
    print("%-5s events=%d  0 jets: %.1f%%  1: %.1f%%  2: %.1f%%  >=3: %.1f%%  mean=%.3f" % (s, len(c), 100*(c == 0).mean(), 100*(c == 1).mean(), 100*(c == 2).mean(), 100*(c >= 3).mean(), c.mean()))
