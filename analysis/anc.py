import numpy as np, uproot
B="/eos/user/p/pkangazi/ZTAG_project/"
def dphi(a,b): return (a-b+np.pi)%(2*np.pi)-np.pi
for s in ("zj","ttZ"):
    res={"all_mass":[],"Zonly_mass":[],"top_pt_frac":[],"n_top":[],"n_all":[]}
    for J in range(300,304):
        t=uproot.open(B+"stages/v5/%s_%d/4_delphes.root"%(s,J))["Delphes"]
        a=t.arrays(["Particle.PID","Particle.Status","Particle.M1","Particle.D1","Particle.Px","Particle.Py","Particle.Pz","Particle.E","Particle.Eta","Particle.Phi"],library="np")
        for i in range(len(a["Particle.PID"])):
            pid=a["Particle.PID"][i]; m1=a["Particle.M1"][i]; d1=a["Particle.D1"][i]
            zs=[k for k in np.where(pid==23)[0] if d1[k]>=0 and pid[d1[k]]!=23]
            if not zs: continue
            k0=zs[0]
            org=np.zeros(len(pid),dtype=np.int8)
            for k in range(len(pid)):
                if abs(pid[k])==23: org[k]=1
                elif abs(pid[k])==6: org[k]=2
                elif m1[k]>=0: org[k]=org[m1[k]]
            px,py,pz,E=[a["Particle."+n][i] for n in ("Px","Py","Pz","E")]
            eta=a["Particle.Eta"][i]; phi=a["Particle.Phi"][i]
            zeta=eta[k0]; zphi=phi[k0]
            fs=(a["Particle.Status"][i]==1)&(~np.isin(np.abs(pid),[12,14,16]))
            m=fs&(np.hypot(eta-zeta,dphi(phi,zphi))<0.8)
            def mass(mm):
                sx,sy,sz,sE=px[mm].sum(),py[mm].sum(),pz[mm].sum(),E[mm].sum()
                return np.sqrt(max(sE*sE-sx*sx-sy*sy-sz*sz,0))
            res["all_mass"].append(mass(m)); res["Zonly_mass"].append(mass(m&(org==1)))
            pt=np.hypot(px,py)
            res["top_pt_frac"].append(pt[m&(org==2)].sum()/max(pt[m].sum(),1e-9))
            res["n_top"].append((m&(org==2)).sum()); res["n_all"].append(m.sum())
    r={k:np.array(v) for k,v in res.items()}
    print("%-4s N=%d | cone mass all: med=%.1f mean=%.1f | Z-origin only: med=%.1f mean=%.1f | top-origin pT frac: mean=%.3f, events with >0: %.1f%% | n_top mean=%.2f of %.1f"%(
      s,len(r["all_mass"]),np.median(r["all_mass"]),r["all_mass"].mean(),np.median(r["Zonly_mass"]),r["Zonly_mass"].mean(),
      r["top_pt_frac"].mean(),100*(r["n_top"]>0).mean(),r["n_top"].mean(),r["n_all"].mean()))
