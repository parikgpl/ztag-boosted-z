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
out={}
for s in ("zj","ttZ"):
    q=[]; dr=[]
    for J in range(300,320):
        for ev in lhe_events(B+"stages/v5/%s_%d/2_madspin.lhe.gz"%(s,J)):
            iz=np.where((ev[:,0]==23)&(ev[:,1]==2))[0]
            if len(iz)==0: continue
            i=iz[0]; z=ev[i]
            d=ev[(ev[:,2]==i+1)&(np.abs(ev[:,0])==5)]
            if len(d)!=2: continue
            p1=np.hypot(d[0,6],d[0,7]); p2=np.hypot(d[1,6],d[1,7])
            zz=min(p1,p2)/(p1+p2)
            q.append(np.sqrt(zz*(1-zz)))
    q=np.array(q); out[s]=q
    print("%-4s N=%d  <sqrt(z(1-z))>=%.4f +- %.4f  median z-proxy=%.3f"%(s,len(q),q.mean(),q.std()/np.sqrt(len(q)),np.median(q)))
print("ratio ttZ/zj = %.3f   (compare: girth ratio 0.168/0.156 = 1.08)"%(out["ttZ"].mean()/out["zj"].mean()))
