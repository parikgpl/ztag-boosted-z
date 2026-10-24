#!/bin/bash
# Build the two C++ executables (tested on lxplus, LCG_106 view).
# usage: bash build.sh   (run from the pipeline/ directory)
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
V=/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt
g++ pythia/shower_lhe.cc -o pythia/shower_lhe -I$PYTHIA8/include $(HepMC3-config --cflags) \
    -L$PYTHIA8/lib -lpythia8 $(HepMC3-config --libs) -ldl
g++ -O2 -std=c++17 fastjet/jetfeat4.cc -o fastjet/jetfeat4 $(root-config --cflags --libs) \
    $(fastjet-config --cxxflags --libs) -L$V/lib -lfastjetcontribfragile
# at run time: export LD_LIBRARY_PATH=$V/lib:$LD_LIBRARY_PATH
