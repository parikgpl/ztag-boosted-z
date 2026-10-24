#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/GenEvent.h"
#include <iostream>
using namespace Pythia8;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: ./shower_lhe in.lhe out.hepmc seed [nmax]\n";
    return 1;
  }
  int nmax = (argc > 4) ? std::atoi(argv[4]) : -1;

  Pythia pythia;
  pythia.readString("Beams:frameType = 4");
  pythia.readString(std::string("Beams:LHEF = ") + argv[1]);
  pythia.readString("Random:setSeed = on");
  pythia.readString(std::string("Random:seed = ") + argv[3]);

  // identical for all four samples
  pythia.readString("PartonLevel:MPI = on");
  pythia.readString("PartonLevel:ISR = on");
  pythia.readString("PartonLevel:FSR = on");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("Next:numberCount = 100");

  if (!pythia.init()) return 1;

  HepMC3::WriterAscii writer(argv[2]);
  Pythia8ToHepMC toHepMC;

  int nOK = 0;
  while (nmax < 0 || nOK < nmax) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      continue;
    }
    HepMC3::GenEvent evt(HepMC3::Units::GEV, HepMC3::Units::MM);
    toHepMC.fill_next_event(pythia, &evt);
    writer.write_event(evt);
    ++nOK;
  }
  writer.close();
  pythia.stat();
  std::cout << "events written: " << nOK << std::endl;
  return 0;
}
