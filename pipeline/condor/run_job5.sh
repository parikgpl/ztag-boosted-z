#!/bin/bash
# usage: run_job.sh SAMPLE(gg|zg|ttbb|ttZ) JOBNUM NEVT
P=$1; J=$2; NEVT=$3
set -e -o pipefail
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
BASE=/eos/home-p/pkangazi/ZTAG_project
case $P in gg) ID=0;; zg) ID=1;; zj) ID=1;; ttbb) ID=2;; ttZ) ID=3;; bbbb) ID=4;; *) echo "bad sample"; exit 1;; esac
SEED=$((ID*10000 + J + 1000))

W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/${P}_${J}
rm -rf $W; mkdir -p $W; cd $W
# f2py wrappers: node system f2py is broken, use numpy.f2py of the LCG python
mkdir -p $W/bin
for n in f2py f2py3 f2py3.11; do
  printf '#!/bin/bash\nexec python3 -m numpy.f2py "$@"\n' > $W/bin/$n; chmod +x $W/bin/$n
done
export PATH=$W/bin:$PATH
trap 'mkdir -p $BASE/fastjet/out/v5/logs; for f in mg shower delphes; do cp $W/$f.log $BASE/fastjet/out/v5/logs/${P}_${J}_$f.log 2>/dev/null; done; true' EXIT

# private copy of the process dir, without old runs
cp -a $BASE/madgraph/MG_$P .
rm -rf MG_$P/Events/run_* MG_$P/RunWeb
# force local multicore running (no sub-jobs to condor from inside the job)
echo "run_mode = 2" >> MG_$P/Cards/me5_configuration.txt
echo "nb_core = 4"  >> MG_$P/Cards/me5_configuration.txt

# MG command file
{
  echo "launch $W/MG_$P"
  if [ $P != gg ] && [ $P != bbbb ]; then
    echo "madspin=ON"
    cp $BASE/madgraph/cards/madspin_$P.dat $W/madspin.dat
    sed -i "1i set seed $SEED" $W/madspin.dat
    echo "$W/madspin.dat"
  fi
  echo "set nevents $NEVT"
  echo "set iseed $SEED"
  echo "set ebeam1 6500"
  echo "set ebeam2 6500"
  case $P in
    gg)  echo "set pt_min_pdg {21: 400}";;
    zg|zj|ttZ) echo "set pt_min_pdg {23: 400}";;
  esac
  echo "done"
} > mg.mg5

mg5_aMC mg.mg5 > mg.log 2>&1
grep "Cross-section" mg.log | tail -n 1 > xsec.txt

if [ $P = gg ] || [ $P = bbbb ]; then
  LHE=MG_$P/Events/run_01/unweighted_events.lhe.gz
else
  LHE=$(ls MG_$P/Events/run_*_decayed_1/unweighted_events.lhe.gz | head -n 1)
fi
zcat $LHE > in.lhe

$BASE/pythia/shower_lhe in.lhe out.hepmc $SEED > shower.log 2>&1
tail -n 1 shower.log

cp $BASE/delphes/delphes_card_CMS.tcl .
if grep -q RandomSeed delphes_card_CMS.tcl; then sed -i "s/^.*RandomSeed.*$/set RandomSeed $SEED/" delphes_card_CMS.tcl; else sed -i "1i set RandomSeed $SEED" delphes_card_CMS.tcl; fi
DelphesHepMC3 delphes_card_CMS.tcl out.root out.hepmc > delphes.log 2>&1

$BASE/fastjet/jetfeat4 out.root jets.root 2>&1 | tail -n 1

OUT=$BASE/fastjet/out/v5
mkdir -p $OUT
cp jets.root $OUT/${P}_${J}_jets2.root
cp xsec.txt  $OUT/${P}_${J}_xsec.txt
mkdir -p $OUT/logs; cp mg.log shower.log delphes.log $OUT/logs/ 2>/dev/null || true
# ---- keep every stage (v5) ----
S=$BASE/stages/v5/${P}_${J}
mkdir -p $S
[ -f MG_$P/Events/run_01/unweighted_events.lhe ] && gzip -c MG_$P/Events/run_01/unweighted_events.lhe > $S/1_madgraph.lhe.gz
[ -f MG_$P/Events/run_01/unweighted_events.lhe.gz ] && cp MG_$P/Events/run_01/unweighted_events.lhe.gz $S/1_madgraph.lhe.gz
ls -l MG_$P/Events/run_01 > $S/mg_run01_listing.txt 2>&1
if [ $P != gg ] && [ $P != bbbb ]; then cp $LHE $S/2_madspin.lhe.gz; fi
gzip -c out.hepmc > $S/3_pythia.hepmc.gz
cp out.root $S/4_delphes.root
cp jets.root $S/5_jets.root
echo "SEED=$SEED used for MG iseed, MadSpin, Pythia, Delphes" > $S/seeds.txt
du -sh $S > $S/sizes.txt; ls -l $S >> $S/sizes.txt
cd /; rm -rf $W
echo "DONE $P $J"
