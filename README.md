# Boosted Z -> b bbar tagger dataset (report + simulation chain)

Four samples at 13 TeV, 20 jobs x 1000 events each: **zj** (pp -> Z j) and **ttZ** (signal, Z -> b bbar),
**bbbb** and **ttbb** (QCD background). In ttZ and ttbb both tops decay leptonically (e or mu).

## Layout
* `report/`    the Persian (XePersian) report: `Report.tex`, chapters, `Report.bib`, `figs/` (all figures are PDF).
* `pipeline/`  everything that produces the data: MadGraph process scripts, MadSpin cards, run/cuts cards,
  the modified `cuts.f` (ttbb, bbbb), Pythia shower program, Delphes CMS card, FastJet jet-feature program,
  HTCondor submit and job scripts.
* `analysis/`  everything after production: HDF5 building, cutflow and origin tables, figure scripts,
  and the stage-by-stage studies of the Z boson (LHE level, polarisation, ancestry).

## Chain (one HTCondor job per sample and job number)
MadGraph 3.5.3 -> MadSpin -> Pythia 8.312 -> Delphes (CMS card) -> FastJet anti-kT R=0.8 (pT > 20 GeV stored).
One random seed per job is used in all stages. The output of every stage is kept in
`stages/v5/<sample>_<job>/` (`1_madgraph.lhe.gz`, `2_madspin.lhe.gz`, `3_pythia.hepmc.gz`, `4_delphes.root`,
`5_jets.root`); jet features go to `fastjet/out/v5/`. The data are not in this repository
(`/eos/user/p/pkangazi/ZTAG_project/{stages,fastjet/out,h5}/v5`).
Job 326 of bbbb produced only 373 events in MadGraph and was replaced by job 340.

## Reproduce
1. `source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh`
2. Create the processes with `pipeline/madgraph/make_*.mg5`, copy the run cards and the modified `cuts.f`
   (ttbb, bbbb) into the process directories, and use the MadSpin cards.
3. Build `pipeline/pythia/shower_lhe.cc` and `pipeline/fastjet/jetfeat4.cc` (see below).
4. From an AFS directory (HTCondor does not accept submit files on /eos): `condor_submit pipeline/condor/ztag5.sub`
   (`jobs_v5.txt` = sample, job number, events).
5. `analysis/cutflow_origin.py` (cutflow and origin figures), `analysis/make_h5.py` (HDF5 dataset),
   `analysis/final_plots*.py`, `analysis/plot_cons.py` (report figures), `analysis/lhe_stages.py`, `pol.py`,
   `zsplit.py`, `anc.py`, `shat_y_zpt.py`, `jet_numbers.py` (numbers and figures of the Z studies).
6. Report: `cd report; xelatex Report.tex; biber Report; xelatex Report.tex; xelatex Report.tex`
   (fonts: B Nazanin, TeX Gyre Termes).

## Building the executables

`pipeline/build.sh` compiles `shower_lhe` (Pythia8 + HepMC3) and `jetfeat4`
(ROOT + FastJet + fjcontrib). Both commands were tested on lxplus with the
LCG_106 view; `jetfeat4` needs `LD_LIBRARY_PATH` to include the view's `lib` directory at run time.

## Job numbering

Job numbers are not contiguous for every sample (e.g. the bbbb jobs are
300-306, 308-310, 312, 313, 316-320, 324-326, with 326 replaced by 340);
use `jobs_v5.txt` rather than a numeric range.

## Report (`report/`)

`report/` contains the Persian report (XePersian LaTeX) that documents this dataset together with its analysis. `Report.pdf` is the compiled version. It describes the generator-level setup of the four samples (zj, ttZ as signal; bbbb, ttbb as background), the event and jet selection with the cutflow, the origin of the selected jets (full Z→bb, partial Z, bb pair, top-decay b, ...), and a comparison of jet-level and constituent-level features between the samples. A dedicated section compares the Z boson in zj and ttZ before and after the MadSpin decay (pT, |eta|, |y|, mass, polarization, energy sharing) to explain the differences in the jet features. All figures are vector PDFs in `report/figs/`, produced by the scripts in `analysis/`.

Files: `Report.tex` (main file, `Report.bib`), `title.tex`, `introduction.tex`, `part1.tex`, `part3.tex`–`part7.tex`, `appendixA.tex`, the logos, and `figs/`.

Build: from inside `report/` run `xelatex -interaction=nonstopmode Report.tex` twice.
