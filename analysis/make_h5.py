"""make_h5.py : build the final HDF5 dataset from fastjet/out/v4/*_jets2.root (made by jetfeat4).
Selection (jet level): pT>400, |eta|<2, nconst>=3  (= has_cons==1) AND origin:
  signal  (zj, ttZ)     : z_both>0 (origin 1, full)  or  z_part>0 (origin 2, partial)          -> label 1
  background (bbbb,ttbb): nb_notop>=2 (origin 1, full) or nb_notop==1 (origin 2, partial), and z_both==0, z_part==0 -> label 0
Full-only dataset = rows with origin==1.
Output (OUT dir):
  events.h5        one row per event (all events, also those without a selected jet)
  jet_features.h5  one row per selected jet; event_id links to events.h5
  constituents.h5  (N_jets, 100, 17) float32, same row order as jet_features.h5 (first 10 = hardest 10)
  images.h5        (N_jets, 32, 32) float32, same row order
usage: python3 make_h5.py [IN_DIR] [OUT_DIR]"""
import sys, os, glob, numpy as np, uproot, h5py

IN  = sys.argv[1] if len(sys.argv) > 1 else "/eos/home-p/pkangazi/ZTAG_project/fastjet/out/v4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/eos/home-p/pkangazi/ZTAG_project/h5/v4"
SAMPLES = ["zj", "ttZ", "bbbb", "ttbb"]            # sample id = index in this list
SIGNAL = {"zj", "ttZ"}
CONV = ["px","py","pz","e","erel","pt","ptrel","eta","etarel","etarot","phi","phirel","phirot","deltaR","costheta","costhetarel","pdgid"]
JETCOLS = ["pt","eta","phi","mass","nconst","girth","nb_notop","nb_top","z_both","z_part","n_hard_g","n_hard_q","d_z","d_top",
           "tau1","tau2","tau3","tau21","tau32","c2","d2","n2","msd","zg","rg"]
NPIX, RANGE, NC = 32, 0.8, 100
os.makedirs(OUT, exist_ok=True)

ev_rows = {k: [] for k in ["event_id","sample","job","evt","n_all_jets","n_saved_jets","z_found","z_pt","z_eta","n_sel_jets"]}
jt_rows = {k: [] for k in ["event_id","sample","job","evt","ijet","label","origin"] + JETCOLS}
cons_list = []
stats = []

for si, s in enumerate(SAMPLES):
    files = sorted(glob.glob(f"{IN}/{s}_3[0-9][0-9]_jets2.root"))
    nfull = npart = nev_tot = 0
    for f in files:
        job = int(os.path.basename(f).split("_")[1])
        with uproot.open(f) as t:
            E = t["evts"].arrays(library="np"); J = t["jets2"].arrays(library="np")
        nev = len(E["evt"])
        if nev < 1000: print("SKIPPED short file", f, nev); continue
        nev_tot += nev
        fin = J["has_cons"] == 1
        if s in SIGNAL:
            full = fin & (J["z_both"] > 0); part = fin & (J["z_both"] == 0) & (J["z_part"] > 0)
        else:
            nofz = (J["z_both"] == 0) & (J["z_part"] == 0)
            full = fin & nofz & (J["nb_notop"] >= 2); part = fin & nofz & (J["nb_notop"] == 1)
        sel = full | part
        origin = np.where(full, 1, 2)[sel].astype(np.int8)
        nfull += int(full.sum()); npart += int(part.sum())
        base = si * 10**7 + job * 10**4
        nsel = np.bincount(J["evt"][sel], minlength=nev)
        ev_rows["event_id"].append(base + E["evt"].astype(np.int64))
        for k, v in [("sample", si), ("job", job)]: ev_rows[k].append(np.full(nev, v, np.int32))
        for k in ["evt","n_all_jets","n_saved_jets","z_found","z_pt","z_eta"]: ev_rows[k].append(E[k])
        ev_rows["n_sel_jets"].append(nsel.astype(np.int32))
        n = int(sel.sum())
        jt_rows["event_id"].append(base + J["evt"][sel].astype(np.int64))
        jt_rows["sample"].append(np.full(n, si, np.int32)); jt_rows["job"].append(np.full(n, job, np.int32))
        jt_rows["evt"].append(J["evt"][sel]); jt_rows["ijet"].append(J["ijet"][sel])
        jt_rows["label"].append(np.full(n, 1 if s in SIGNAL else 0, np.int8)); jt_rows["origin"].append(origin)
        for k in JETCOLS: jt_rows[k].append(J[k][sel])
        cons_list.append(np.stack([J["c_" + k][sel] for k in CONV], axis=2).astype(np.float32))   # (n,100,17)
    stats.append((s, len(files), nev_tot, nfull, npart))

ev = {k: np.concatenate(v) for k, v in ev_rows.items()}
jt = {k: np.concatenate(v) for k, v in jt_rows.items()}
cons = np.concatenate(cons_list, axis=0); del cons_list
N = len(jt["pt"])
assert cons.shape == (N, NC, len(CONV)), cons.shape
bad = int((~np.isfinite(cons)).sum()) + sum(int((~np.isfinite(v.astype(float))).sum()) for v in jt.values())
print("non-finite values:", bad)

# image: pT-weighted (etarel, phirel) map, pixel = sum pT(particle)/pT(jet), all 100 stored constituents
ipt, ie, ip = CONV.index("ptrel"), CONV.index("etarel"), CONV.index("phirel")
img = np.zeros((N, NPIX, NPIX), np.float32)
for a in range(0, N, 5000):
    c = cons[a:a+5000]; m = c.shape[0]
    pr = c[:, :, ipt]
    ix = np.floor((c[:, :, ie] + RANGE) / (2 * RANGE) * NPIX).astype(int)
    iy = np.floor((c[:, :, ip] + RANGE) / (2 * RANGE) * NPIX).astype(int)
    ok = (pr > 0) & (ix >= 0) & (ix < NPIX) & (iy >= 0) & (iy < NPIX)
    flat = (np.arange(m)[:, None] * NPIX * NPIX + ix * NPIX + iy)[ok]
    img[a:a+m] = np.bincount(flat, weights=pr[ok], minlength=m * NPIX * NPIX).reshape(m, NPIX, NPIX)

def write(name, d, attrs=None):
    with h5py.File(os.path.join(OUT, name), "w") as h:
        for k, v in d.items(): h.create_dataset(k, data=v, compression="gzip")
        for k, v in (attrs or {}).items(): h.attrs[k] = v
meta = {"samples": SAMPLES, "origin": "1=full (signal: both b of Z in jet; background: >=2 non-top b), 2=partial (signal: Z axis in cone, <2 of its b; background: exactly 1 non-top b)", "label": "1=signal (zj,ttZ), 0=background (bbbb,ttbb)"}
write("events.h5", ev, meta); write("jet_features.h5", jt, meta)
with h5py.File(os.path.join(OUT, "constituents.h5"), "w") as h:
    d = h.create_dataset("constituents", data=cons, chunks=(1000, NC, len(CONV)), compression="gzip")
    d.attrs["features"] = CONV; d.attrs["note"] = "sorted by pT, zero padded to 100; first 10 = the 10 hardest; row i = row i of jet_features.h5"
with h5py.File(os.path.join(OUT, "images.h5"), "w") as h:
    d = h.create_dataset("images", data=img, chunks=(1000, NPIX, NPIX), compression="gzip")
    d.attrs["axes"] = "etarel (x), phirel (y); range +-%.1f; pixel = sum pT/pT_jet" % RANGE

print("\nsample  files  events   full   partial   (final jets)")
for s, nf, nev, a, b in stats: print(f"{s:6s} {nf:5d} {nev:7d} {a:7d} {b:8d}")
print("events:", len(ev["event_id"]), "| jets:", N, "| cons:", cons.shape, "| images:", img.shape)
print("events with >=1 selected jet per sample:", {s: round(float((ev['n_sel_jets'][ev['sample'] == i] > 0).mean()), 3) for i, s in enumerate(SAMPLES)})
print("image sum / sum ptrel of first 100 cons (should be ~equal):", float(img.sum()), float(np.clip(cons[:, :, ipt], 0, None).sum()))
print("written to", OUT)
