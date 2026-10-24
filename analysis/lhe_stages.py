import gzip, os, numpy as np
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
B="/eos/user/p/pkangazi/ZTAG_project/"
OUT="lhe_figs"; os.makedirs(OUT,exist_ok=True)
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
def kin(px,py,pz,E):
    pt=np.hypot(px,py); p=np.sqrt(px*px+py*py+pz*pz)
    eta=np.arctanh(np.clip(pz/max(p,1e-9),-0.999999999,0.999999999))
    y=0.5*np.log((E+pz)/max(E-pz,1e-9)); return pt,eta,y,np.arctan2(py,px)
def zvars(z):
    pt,eta,y,_=kin(z[6],z[7],z[8],z[9])
    return dict(Z_pt=pt,Z_abs_eta=abs(eta),Z_abs_y=abs(y),Z_mass=np.sqrt(max(z[9]**2-z[6]**2-z[7]**2-z[8]**2,0)))
R={}
def add(key,d):
    for k,v in d.items(): R.setdefault(key,{}).setdefault(k,[]).append(v)
for s in ("zj","ttZ"):
    for J in range(300,320):
        d=B+"stages/v5/%s_%d/"%(s,J); print(s,J,flush=True)
        for ev in lhe_events(d+"1_madgraph.lhe.gz"):
            z=ev[ev[:,0]==23]
            if len(z): add((s,"MG"),zvars(z[0]))
        for ev in lhe_events(d+"2_madspin.lhe.gz"):
            iz=np.where((ev[:,0]==23)&(ev[:,1]==2))[0]
            if len(iz)==0: continue
            i=iz[0]; z=ev[i]
            dd=ev[(ev[:,2]==i+1)&(np.abs(ev[:,0])==5)]
            b=dd[dd[:,0]==5]; bb=dd[dd[:,0]==-5]
            if len(b)!=1 or len(bb)!=1: continue
            b=b[0]; bb=bb[0]; v=zvars(z)
            _,e1,_,p1=kin(*b[6:10]); _,e2,_,p2=kin(*bb[6:10])
            q1=np.hypot(b[6],b[7]); q2=np.hypot(bb[6],bb[7]); zz=min(q1,q2)/(q1+q2)
            pz=z[6:9]; E=z[9]; bet=pz/E; b2=bet@bet; g=1/np.sqrt(1-b2)
            vv=b[6:9]; vp=vv+((g-1)*(vv@bet)/b2-g*b[9])*bet
            v.update(dR_bb=np.hypot(e1-e2,(p1-p2+np.pi)%(2*np.pi)-np.pi),pt_b_lead=max(q1,q2),pt_b_sub=min(q1,q2),
                     z_split=zz,sqrt_z1mz=np.sqrt(zz*(1-zz)),cos_theta_star=vp@pz/(np.linalg.norm(vp)*np.linalg.norm(pz)))
            add((s,"MS"),v)
A={k:{kk:np.array(vv) for kk,vv in d.items()} for k,d in R.items()}
COL={"zj":"tab:purple","ttZ":"tab:orange"}; NAME={"zj":"Z+jet","ttZ":r"$t\bar tZ$"}
def hist(ax,x,rng,nb,label,c,ls):
    h,e=np.histogram(np.clip(x,rng[0],rng[1]-1e-9),bins=nb,range=rng); d=h/h.sum()/np.diff(e)
    ax.step(e,np.append(d,d[-1]),where="post",color=c,ls=ls,lw=1.8,label=label)
ZV=[("Z_pt",r"Z $p_T$ [GeV]",(400,1000),40,False),("Z_abs_eta",r"Z $|\eta|$",(0,3),30,False),
    ("Z_abs_y",r"Z $|y|$",(0,3),30,False),("Z_mass","Z mass [GeV]",(70,110),80,True)]
DV=[("dR_bb",r"$\Delta R(b,\bar b)$",(0,1.5),40,False),("pt_b_lead",r"leading b $p_T$ [GeV]",(0,1000),40,False),
    ("pt_b_sub",r"subleading b $p_T$ [GeV]",(0,500),40,False),("z_split",r"$z=p_T^{sub}/(p_T^{b}+p_T^{\bar b})$",(0,0.5),25,False),
    ("sqrt_z1mz",r"$\sqrt{z(1-z)}$",(0,0.5),25,False),("cos_theta_star",r"$\cos\theta^*$ (b in Z rest frame)",(-1,1),20,False)]
pdf=PdfPages(OUT+"/lhe_stages_all.pdf")
for var,xl,rng,nb,logy in ZV:
    fig,ax=plt.subplots(figsize=(7,5))
    for s in ("zj","ttZ"):
        for st,ls in (("MG","--"),("MS","-")):
            hist(ax,A[(s,st)][var],rng,nb,"%s %s"%(NAME[s],"MadGraph (before decay)" if st=="MG" else "MadSpin (after decay)"),COL[s],ls)
    ax.set_xlabel(xl); ax.set_ylabel("normalised to unit area"); ax.legend(frameon=False,fontsize=9)
    if logy: ax.set_yscale("log")
    fig.savefig(OUT+"/lhe_"+var+".pdf",bbox_inches="tight"); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
for var,xl,rng,nb,logy in DV:
    fig,ax=plt.subplots(figsize=(7,5))
    for s in ("zj","ttZ"):
        x=A[(s,"MS")][var]; hist(ax,x,rng,nb,"%s (N=%d)"%(NAME[s],len(x)),COL[s],"-")
    if var=="cos_theta_star":
        t=[]
        for s in ("zj","ttZ"):
            x=A[(s,"MS")][var]; t.append("%s: $f_L$=%.2f"%(NAME[s],2-5*np.mean(x**2)))
        ax.text(0.03,0.04,"\n".join(t),transform=ax.transAxes,fontsize=10)
    ax.set_xlabel(xl); ax.set_ylabel("normalised to unit area"); ax.legend(frameon=False,fontsize=10)
    fig.savefig(OUT+"/lhe_"+var+".pdf",bbox_inches="tight"); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
pdf.close()
for (s,st),d in sorted(A.items()):
    print(s,st,"N=%d"%len(d["Z_pt"])," ".join("%s: mean=%.3f"%(k,v.mean()) for k,v in d.items()))
print("figures in",OUT)
