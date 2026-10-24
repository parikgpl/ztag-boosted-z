// jetfeat4: ALL anti-kT R=0.8 jets (pT>PTSAVE) of every event, with truth labels, jet substructure
// and the 17 features of the NCONS hardest constituents of every jet that passes the analysis cuts.
// usage: ./jetfeat4 in_delphes.root out.root [nmax]
// tree "evts": one row per event.  tree "jets2": one row per stored jet (rows of one event are consecutive,
// ordered by pT; "evt" links a jet to its event and "ijet" is its pT rank inside the event).
// Constituent arrays c_<name>[NCONS] are filled for jets with pT>400, |eta|<2, nconst>=3 (has_cons=1), else 0.
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/EnergyCorrelator.hh"
#include "fastjet/contrib/SoftDrop.hh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
using namespace fastjet;

static const int NCONS = 100;   // constituents stored per jet (the tagger uses the first 10, the image uses all)
static const int NF = 17;       // features per constituent
static float cf[NF][NCONS];

static double wrapPhi(double d) { while (d > M_PI) d -= 2*M_PI; while (d < -M_PI) d += 2*M_PI; return d; }
static double dRf(double e1, double p1, double e2, double p2) {
  double dp = wrapPhi(p1 - p2);
  return sqrt((e1-e2)*(e1-e2) + dp*dp);
}
// pid is kept in user_index: track = Delphes PID (signed), photon tower = 22, neutral hadron tower = 130
static void addP(vector<PseudoJet>& v, double pt, double eta, double phi, int pid) {
  if (pt <= 0) return;
  v.emplace_back(pt*cos(phi), pt*sin(phi), pt*sinh(eta), pt*cosh(eta));
  v.back().set_user_index(pid);
}

int main(int argc, char** argv) {
  if (argc < 3) { cerr << "usage: ./jetfeat4 in_delphes.root out.root [nmax]\n"; return 1; }
  long nmax = (argc > 3) ? atol(argv[3]) : -1;
  const double R = 0.8, PTSAVE = 20.;

  TFile fin(argv[1]);
  TTreeReader rd("Delphes", &fin);
  TTreeReaderArray<float> tPT(rd,"EFlowTrack.PT"), tEta(rd,"EFlowTrack.Eta"), tPhi(rd,"EFlowTrack.Phi");
  TTreeReaderArray<int>   tPID(rd,"EFlowTrack.PID");
  TTreeReaderArray<float> pET(rd,"EFlowPhoton.ET"), pEta(rd,"EFlowPhoton.Eta"), pPhi(rd,"EFlowPhoton.Phi");
  TTreeReaderArray<float> nET(rd,"EFlowNeutralHadron.ET"), nEta(rd,"EFlowNeutralHadron.Eta"), nPhi(rd,"EFlowNeutralHadron.Phi");
  TTreeReaderArray<int>   gPID(rd,"Particle.PID"), gD1(rd,"Particle.D1"), gD2(rd,"Particle.D2");
  TTreeReaderArray<int>   gM1(rd,"Particle.M1"), gSt(rd,"Particle.Status");
  TTreeReaderArray<float> gEta(rd,"Particle.Eta"), gPhi(rd,"Particle.Phi"), gPT(rd,"Particle.PT");

  TFile fout(argv[2], "RECREATE");
  TTree tj("jets2", "all jets with truth, substructure and constituents");
  TTree te("evts", "per event");
  int evt = 0, ijet = 0, nconst = 0, nbn = 0, nbt = 0, zboth = 0, zpart = 0, nglu = 0, nlq = 0, hascons = 0;
  float pt = 0, eta = 0, phi = 0, mass = 0, girth = 0, dz = 99, dtop = 99;
  tj.Branch("evt", &evt, "evt/I");
  tj.Branch("ijet", &ijet, "ijet/I");          // pT rank of the jet inside its event (0 = hardest)
  tj.Branch("pt", &pt, "pt/F");        tj.Branch("eta", &eta, "eta/F");
  tj.Branch("phi", &phi, "phi/F");     tj.Branch("mass", &mass, "mass/F");
  tj.Branch("nconst", &nconst, "nconst/I"); tj.Branch("girth", &girth, "girth/F");
  tj.Branch("nb_notop", &nbn, "nb_notop/I");   // b quarks not from a top, inside the cone
  tj.Branch("nb_top", &nbt, "nb_top/I");       // b quarks from a top, inside the cone
  tj.Branch("z_both", &zboth, "z_both/I");     // jet contains the Z and both its b quarks
  tj.Branch("z_part", &zpart, "z_part/I");     // Z inside the cone but not both b's
  tj.Branch("n_hard_g", &nglu, "n_hard_g/I");  // hard-process gluons (status 23) inside the cone
  tj.Branch("n_hard_q", &nlq, "n_hard_q/I");   // hard-process light quarks u d s c (status 23)
  tj.Branch("d_z", &dz, "d_z/F");              // dR(jet, truth Z), 99 if no Z
  tj.Branch("d_top", &dtop, "d_top/F");        // dR(jet, nearest truth top), 99 if none
  float tau1=-1, tau2=-1, tau3=-1, tau21=-1, tau32=-1, c2=-1, d2=-1, n2=-1, msd=-1, zg=-1, rg=-1;
  tj.Branch("tau1",&tau1,"tau1/F"); tj.Branch("tau2",&tau2,"tau2/F"); tj.Branch("tau3",&tau3,"tau3/F");
  tj.Branch("tau21",&tau21,"tau21/F"); tj.Branch("tau32",&tau32,"tau32/F");
  tj.Branch("c2",&c2,"c2/F"); tj.Branch("d2",&d2,"d2/F"); tj.Branch("n2",&n2,"n2/F");
  tj.Branch("msd",&msd,"msd/F"); tj.Branch("zg",&zg,"zg/F"); tj.Branch("rg",&rg,"rg/F");
  tj.Branch("has_cons", &hascons, "has_cons/I");   // 1 if the constituent arrays below are filled

  const char* cn[NF] = {"px","py","pz","e","erel","pt","ptrel","eta","etarel","etarot",
                        "phi","phirel","phirot","deltaR","costheta","costhetarel","pdgid"};
  for (int k = 0; k < NF; k++) {
    char nm[64], tp[64];
    snprintf(nm, 64, "c_%s", cn[k]); snprintf(tp, 64, "c_%s[%d]/F", cn[k], NCONS);
    tj.Branch(nm, cf[k], tp);
  }

  contrib::Nsubjettiness NS1(1, contrib::KT_Axes(), contrib::NormalizedMeasure(1.0, R));
  contrib::Nsubjettiness NS2(2, contrib::KT_Axes(), contrib::NormalizedMeasure(1.0, R));
  contrib::Nsubjettiness NS3(3, contrib::KT_Axes(), contrib::NormalizedMeasure(1.0, R));
  contrib::EnergyCorrelatorC2 C2f(1.0);
  contrib::EnergyCorrelatorD2 D2f(1.0);
  contrib::EnergyCorrelatorN2 N2f(1.0);
  contrib::SoftDrop SD(0.0, 0.1);
  int nall = 0, nsaved = 0, nzfound = 0;
  float zpt = -1, zeta = -99;
  te.Branch("evt", &evt, "evt/I");
  te.Branch("n_all_jets", &nall, "n_all_jets/I");   // every jet from the clustering (no pT cut)
  te.Branch("n_saved_jets", &nsaved, "n_saved_jets/I");   // jets with pT>PTSAVE = rows in jets2 for this event
  te.Branch("z_found", &nzfound, "z_found/I");
  te.Branch("z_pt", &zpt, "z_pt/F");  te.Branch("z_eta", &zeta, "z_eta/F");

  JetDefinition jd(antikt_algorithm, R);
  long nev = 0;
  while (rd.Next()) {
    if (nmax >= 0 && nev >= nmax) break;
    evt = nev++;

    vector<PseudoJet> parts;
    for (size_t i = 0; i < tPT.GetSize(); i++) addP(parts, tPT[i], tEta[i], tPhi[i], tPID[i]);
    for (size_t i = 0; i < pET.GetSize(); i++) addP(parts, pET[i], pEta[i], pPhi[i], 22);
    for (size_t i = 0; i < nET.GetSize(); i++) addP(parts, nET[i], nEta[i], nPhi[i], 130);

    int ng = gPID.GetSize();
    int zi = -1, b1 = -1, b2 = -1;
    for (int i = 0; i < ng; i++) {
      if (gPID[i] != 23) continue;
      int a = gD1[i], b = gD2[i];
      if (a < 0 || b < 0 || a >= ng || b >= ng) continue;
      if (abs(gPID[a]) == 5 && abs(gPID[b]) == 5) { zi = i; b1 = a; b2 = b; }
    }
    zpt = -1; zeta = -99; nzfound = (zi >= 0);
    if (zi >= 0) { zpt = gPT[zi]; zeta = gEta[zi]; }

    vector<int> bNo, bTp, hg, hq, tops;
    for (int i = 0; i < ng; i++) {
      int id = abs(gPID[i]);
      if (id == 6) {   // last copy of the top
        bool last = true;
        int d1 = gD1[i], d2 = gD2[i];
        if (d1 >= 0 && d2 >= d1)
          for (int k = d1; k <= d2 && k < ng; k++) if (gPID[k] == gPID[i]) { last = false; break; }
        if (last) tops.push_back(i);
      }
      if (gSt[i] == 23) {
        if (id == 21) hg.push_back(i);
        else if (id >= 1 && id <= 4) hq.push_back(i);
      }
      if (id != 5) continue;
      bool last = true;
      int d1 = gD1[i], d2 = gD2[i];
      if (d1 >= 0 && d2 >= d1)
        for (int k = d1; k <= d2 && k < ng; k++) if (gPID[k] == gPID[i]) { last = false; break; }
      if (!last) continue;
      int a = gM1[i], guard = 0;
      while (a >= 0 && a < ng && abs(gPID[a]) == 5 && guard++ < 100) a = gM1[a];
      bool fromTop = (a >= 0 && a < ng && abs(gPID[a]) == 6);
      (fromTop ? bTp : bNo).push_back(i);
    }

    ClusterSequence cs(parts, jd);
    vector<PseudoJet> all = sorted_by_pt(cs.inclusive_jets(0.));
    nall = all.size();
    nsaved = 0;
    for (const PseudoJet& jet : all) if (jet.pt() >= PTSAVE) nsaved++;
    te.Fill();
    ijet = 0;
    for (const PseudoJet& jet : all) {
      if (jet.pt() < PTSAVE) break;              // sorted by pT
      double jeta = jet.eta(), jphi = jet.phi_std();
      pt = jet.pt(); eta = jeta; phi = jphi; mass = jet.m();
      nbn = 0; nbt = 0; nglu = 0; nlq = 0; zboth = 0; zpart = 0; dz = 99; dtop = 99;
      for (int i : bNo) if (dRf(jeta, jphi, gEta[i], gPhi[i]) < R) nbn++;
      for (int i : bTp) if (dRf(jeta, jphi, gEta[i], gPhi[i]) < R) nbt++;
      for (int i : hg)  if (dRf(jeta, jphi, gEta[i], gPhi[i]) < R) nglu++;
      for (int i : hq)  if (dRf(jeta, jphi, gEta[i], gPhi[i]) < R) nlq++;
      for (int i : tops) dtop = min(dtop, (float)dRf(jeta, jphi, gEta[i], gPhi[i]));
      if (zi >= 0) {
        dz = dRf(jeta, jphi, gEta[zi], gPhi[zi]);
        if (dz < R) {
          int nin = (dRf(jeta, jphi, gEta[b1], gPhi[b1]) < R) + (dRf(jeta, jphi, gEta[b2], gPhi[b2]) < R);
          if (nin == 2) zboth = 1; else zpart = 1;
        }
      }
      vector<PseudoJet> cons = sorted_by_pt(jet.constituents());   // hardest first
      nconst = cons.size();
      double sumpt = 0, g = 0;
      for (const PseudoJet& c : cons) sumpt += c.pt();
      for (const PseudoJet& c : cons) g += c.pt() * dRf(c.eta(), c.phi_std(), jeta, jphi);
      girth = sumpt > 0 ? g / sumpt : 0;
      tau1=tau2=tau3=tau21=tau32=c2=d2=n2=msd=zg=rg=-1;
      hascons = 0;
      for (int k = 0; k < NF; k++) for (int i = 0; i < NCONS; i++) cf[k][i] = 0;
      if (pt > 400. && fabs(eta) < 2.0 && nconst >= 3) {   // substructure and constituents only for jets passing the cuts
        tau1 = NS1(jet); tau2 = NS2(jet); tau3 = NS3(jet);
        tau21 = tau1 > 0 ? tau2/tau1 : -1;  tau32 = tau2 > 0 ? tau3/tau2 : -1;
        c2 = C2f(jet); d2 = D2f(jet); n2 = N2f(jet);
        PseudoJet sj = SD(jet);
        if (sj != 0) { msd = sj.m(); zg = sj.structure_of<contrib::SoftDrop>().symmetry(); rg = sj.structure_of<contrib::SoftDrop>().delta_R(); }
        else msd = 0;

        // 17 features of every constituent (up to NCONS), hardest first.
        // rotated frame: hardest constituent at the origin, second hardest on the negative phi axis.
        hascons = 1;
        double e0 = cons[0].eta(), p0 = cons[0].phi_std(), th = 0;
        if (cons.size() > 1) {
          double a1 = cons[1].eta() - e0, b1r = wrapPhi(cons[1].phi_std() - p0);
          if (a1 != 0 || b1r != 0) th = -M_PI/2 - atan2(b1r, a1);
        }
        for (size_t i = 0; i < cons.size() && (int)i < NCONS; i++) {
          const PseudoJet& q = cons[i];
          double ce = q.eta(), cp = q.phi_std(), x = ce - e0, y = wrapPhi(cp - p0);
          double er = ce - jeta, pr = wrapPhi(cp - jphi);
          double cosrel = (q.px()*jet.px() + q.py()*jet.py() + q.pz()*jet.pz()) / (q.modp()*jet.modp());
          double v[NF] = { q.px(), q.py(), q.pz(), q.E(), q.E()/jet.E(), q.pt(), q.pt()/jet.pt(),
                           ce, er, x*cos(th) - y*sin(th),
                           cp, pr, x*sin(th) + y*cos(th),
                           sqrt(er*er + pr*pr), q.pz()/q.modp(), cosrel, (double)q.user_index() };
          for (int k = 0; k < NF; k++) cf[k][i] = (float)v[k];
        }
      }
      tj.Fill();
      ijet++;
    }
  }
  fout.cd(); tj.Write(); te.Write(); fout.Close();
  cout << "events written: " << nev << " jets rows: " << tj.GetEntries() << endl;
  return 0;
}
