import sys, gzip, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="/eos/user/p/pkangazi/ZTAG_project/"
NJ=int(sys.argv[1]) if len(sys.argv)>1 else 4
def lhe_events(path):
    ev=None; hdr=False
    with gzip.open(path,"rt") as f:
        for line in f:
            if line.startswith("<event>"): ev=[]; hdr=True; continue
            if line.startswith("</event>"):
                yield np.array(ev); ev=None; continue
            if ev is None: continue
            if hdr: hdr=False; continue
            if line[0] in "<#": continue
            t=line.split()
            if len(t)>=13: ev.append([float(x) for x in t[:13]])
def kin(px,py,pz,E):
    pt=np.hypot(px,py); p=np.sqrt(px*px+py*py+pz*pz)
    eta=np.arctanh(np.clip(pz/np.maximum(p,1e-9),-0.999999999,0.999999999))
    y=0.5*np.log((E+pz)/np.maximum(E-pz,1e-9))
    return pt,eta,y,np.arctan2(py,px)
def dphi(a,b):
    d=a-b; return (d+np.pi)%(2*np.pi)-np.pi
R={}
def add(s,st,k,v): R.setdefault(s,{}).setdefault(st,{}).setdefault(k,[]).append(v)
for s in ("zj","ttZ"):
    for J in range(300,300+NJ):
        d=B+"stages/v5/%s_%d/"%(s,J)
        print(s,J,flush=True)
        # stage 1
        for ev in lhe_events(d+"1_madgraph.lhe.gz"):
            z=ev[ev[:,0]==23]
            if len(z)==0: continue
            z=z[0]; pt,eta,y,_=kin(z[6],z[7],z[8],z[9])
            add(s,"1_MG","Z_pt",pt); add(s,"1_MG","Z_abs_eta",abs(eta)); add(s,"1_MG","Z_abs_y",abs(y))
            fs=ev[(ev[:,1]==1)&(ev[:,0]!=23)]
            if len(fs): add(s,"1_MG","recoil_pt",np.hypot(fs[:,6],fs[:,7]).max())
        # stage 2
        for ev in lhe_events(d+"2_madspin.lhe.gz"):
            iz=np.where((ev[:,0]==23)&(ev[:,1]==2))[0]
            if len(iz)==0: continue
            i=iz[0]; z=ev[i]
            dau=ev[(ev[:,2]==i+1)&(np.abs(ev[:,0])==5)]
            if len(dau)!=2: continue
            b=dau[dau[:,0]==5]; bb=dau[dau[:,0]==-5]
            if len(b)!=1 or len(bb)!=1: continue
            b=b[0]; bb=bb[0]
            _,e1,_,p1=kin(*b[6:10]); _,e2,_,p2=kin(*bb[6:10])
            add(s,"2_MS","dR_bb",np.hypot(e1-e2,dphi(p1,p2)))
            add(s,"2_MS","pt_b_lead",max(np.hypot(b[6],b[7]),np.hypot(bb[6],bb[7])))
            add(s,"2_MS","pt_b_sub",min(np.hypot(b[6],b[7]),np.hypot(bb[6],bb[7])))
            pz=np.array(z[6:9]); E=z[9]; bet=pz/E; b2=bet@bet; g=1/np.sqrt(1-b2)
            v=np.array(b[6:9]); bp=v@bet
            vp=v+((g-1)*bp/b2-g*b[9])*bet
            add(s,"2_MS","abs_cos_theta_star",abs(vp@pz/(np.linalg.norm(vp)*np.linalg.norm(pz))))
        # stage 3 (Pythia record stored in Delphes file)
        br=["Particle.PID","Particle.Status","Particle.D1","Particle.Px","Particle.Py","Particle.Pz","Particle.E","Particle.Eta","Particle.Phi"]
        t=uproot.open(d+"4_delphes.root")["Delphes"]
        a=t.arrays(br,library="np")
        for i in range(len(a["Particle.PID"])):
            pid=a["Particle.PID"][i]; d1=a["Particle.D1"][i]
            zs=[k for k in np.where(pid==23)[0] if d1[k]>=0 and pid[d1[k]]!=23]
            if not zs: continue
            k=zs[0]
            px,py,pz,E=[a["Particle."+n][i] for n in ("Px","Py","Pz","E")]
            zpt,zeta,_,zphi=kin(px[k],py[k],pz[k],E[k])
            fs=(a["Particle.Status"][i]==1)&(~np.isin(np.abs(pid),[12,14,16]))
            dr=np.hypot(a["Particle.Eta"][i]-zeta,dphi(a["Particle.Phi"][i],zphi))
            m=fs&(dr<0.8)
            sx,sy,sz,sE=px[m].sum(),py[m].sum(),pz[m].sum(),E[m].sum()
            add(s,"3_Pythia","n_in_cone",m.sum())
            add(s,"3_Pythia","cone_pt_over_Zpt",np.hypot(sx,sy)/zpt)
            add(s,"3_Pythia","cone_mass",np.sqrt(max(sE*sE-sx*sx-sy*sy-sz*sz,0)))
        # stage 5
        j=uproot.open(d+"5_jets.root")["jets2"].arrays(library="np")
        sel=j["z_both"]==1
        for key in ("mass","nconst","girth"):
            if key in j: add(s,"5_Jets","%s(z_both)"%key,j[key][sel])
        add(s,"5_Jets","frac_jets_z_both",np.array([sel.mean()]))
out=open("stagecmp_table.txt","w")
def flat(l): return np.concatenate([np.atleast_1d(x) for x in l]).astype(float)
panels=[]
for st in sorted(R["zj"]):
    for k in R["zj"][st]:
        if k not in R["ttZ"].get(st,{}): continue
        a=flat(R["zj"][st][k]); b=flat(R["ttZ"][st][k]); panels.append((st,k,a,b))
        line="%-9s %-24s zj: N=%6d mean=%9.3f med=%9.3f [16,84]=[%8.3f,%8.3f] | ttZ: N=%6d mean=%9.3f med=%9.3f [16,84]=[%8.3f,%8.3f]"%(
            st,k,len(a),a.mean(),np.median(a),*np.percentile(a,[16,84]),len(b),b.mean(),np.median(b),*np.percentile(b,[16,84]))
        print(line); out.write(line+"\n")
out.close()
n=len(panels); c=4; r=(n+c-1)//c
fig,ax=plt.subplots(r,c,figsize=(4*c,3*r)); ax=ax.ravel()
for q,(st,k,a,b) in enumerate(panels):
    lo,hi=np.percentile(np.concatenate([a,b]),[0.5,99.5])
    if hi<=lo: hi=lo+1
    ax[q].hist(a,bins=40,range=(lo,hi),density=True,histtype="step",label="zj")
    ax[q].hist(b,bins=40,range=(lo,hi),density=True,histtype="step",label="ttZ")
    ax[q].set_title(st+" "+k,fontsize=9)
ax[0].legend()
for q in range(n,len(ax)): ax[q].axis("off")
plt.tight_layout(); plt.savefig("stagecmp.png",dpi=110)
print("saved stagecmp.png")
