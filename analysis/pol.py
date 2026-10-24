import gzip, numpy as np
B="/eos/user/p/pkangazi/ZTAG_project/"
def lhe_events(path):
    ev=None; hdr=False
    with gzip.open(path,"rt") as f:
        for line in f:
            if line.startswith("<event>"): ev=[]; hdr=True; continue
            if line.startswith("</event>"): yield np.array(ev); ev=None; continue
            if ev is None: continue
            if hdr: hdr=False; continue
            if line[0] in "<#": continue
            t=line.split()
            if len(t)>=13: ev.append([float(x) for x in t[:13]])
def costar(ev):
    iz=np.where((ev[:,0]==23)&(ev[:,1]==2))[0]
    if len(iz)==0: return None
    i=iz[0]; z=ev[i]
    d=ev[(ev[:,2]==i+1)&(ev[:,0]==5)]
    if len(d)!=1: return None
    b=d[0]; pz=z[6:9]; E=z[9]; bet=pz/E; b2=bet@bet; g=1/np.sqrt(1-b2)
    v=b[6:9]; vp=v+((g-1)*(v@bet)/b2-g*b[9])*bet
    return vp@pz/(np.linalg.norm(vp)*np.linalg.norm(pz)), np.hypot(pz[0],pz[1])
for s in ("zj","ttZ"):
    x=[]; pt=[]
    for J in range(300,320):
        for ev in lhe_events(B+"stages/v5/%s_%d/2_madspin.lhe.gz"%(s,J)):
            r=costar(ev)
            if r: x.append(r[0]); pt.append(r[1])
    x=np.array(x); pt=np.array(pt); x2=x*x; N=len(x)
    fL=2-5*x2.mean(); err=5*x2.std()/np.sqrt(N)
    print("%-4s N=%6d <cos^2>=%.4f  f_L=%.3f +- %.3f   <cos>=%.3f"%(s,N,x2.mean(),fL,err,x.mean()))
    h,_=np.histogram(x,bins=10,range=(-1,1),density=True)
    print("     cos*-hist (10 bins -1..1):"," ".join("%.2f"%v for v in h))
    for lo,hi in ((400,450),(450,550),(550,2000)):
        m=(pt>=lo)&(pt<hi)
        if m.sum()>200: print("     pT[%d,%d): N=%d f_L=%.3f +- %.3f"%(lo,hi,m.sum(),2-5*x2[m].mean(),5*x2[m].std()/np.sqrt(m.sum())))
