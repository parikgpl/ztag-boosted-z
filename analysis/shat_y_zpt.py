"""Hard-process kinematics from the MadGraph-level LHE of every sample: sqrt(shat), |y| of the
hard system, and the Z pT tail (zj, ttZ). usage: python3 shat_y_zpt.py"""
import gzip, numpy as np, glob
B = "/eos/user/p/pkangazi/ZTAG_project/stages/v5/"
def evs(path):
    ev = None; hdr = False
    for line in gzip.open(path, "rt"):
        if line.startswith("<event>"): ev = []; hdr = True; continue
        if line.startswith("</event>"): yield np.array(ev); ev = None; continue
        if ev is None: continue
        if hdr: hdr = False; continue
        if line[0] in "<#": continue
        t = line.split()
        if len(t) >= 13: ev.append([float(x) for x in t[:13]])
for s in ("zj", "ttZ", "bbbb", "ttbb"):
    files = sorted(glob.glob(B + "%s_*/1_madgraph.lhe.gz" % s)); m = []; y = []; zpt = []
    for f in files:
        for ev in evs(f):
            P = ev[ev[:, 1] == 1][:, 6:10].sum(0)
            m.append(np.sqrt(max(P[3]**2 - P[0]**2 - P[1]**2 - P[2]**2, 0)))
            y.append(abs(0.5 * np.log((P[3] + P[2]) / (P[3] - P[2]))))
            z = ev[ev[:, 0] == 23]
            if len(z): zpt.append(np.hypot(z[0][6], z[0][7]))
    m = np.array(m); y = np.array(y)
    line = "%-5s jobs=%d N=%d sqrt(shat) median=%.0f | |y_sys| mean=%.3f" % (s, len(files), len(m), np.median(m), y.mean())
    if zpt:
        zpt = np.array(zpt)
        line += " | Z pT mean=%.1f median=%.1f frac(>600)=%.3f frac(>700)=%.3f frac(>800)=%.3f" % (zpt.mean(), np.median(zpt), (zpt > 600).mean(), (zpt > 700).mean(), (zpt > 800).mean())
    print(line)
