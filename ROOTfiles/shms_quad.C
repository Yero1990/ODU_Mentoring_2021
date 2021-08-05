#include <TSystem.h>
#include <TString.h>
#include "TFile.h"
#include "TTree.h"
#include <TNtuple.h>
#include "TCanvas.h"
#include <iostream>
#include <fstream>
#include "TMath.h"
#include "TH1F.h"
#include <TH2.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TROOT.h>
#include <TMath.h>
#include <TLegend.h>
#include <TPaveLabel.h>
#include <TProfile.h>
#include <TObjArray.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include<math.h>
using namespace std;

/*
  This script reads an input file from optics data, and outputs 
  .pdf and/or 2D focal plane histogram objects.
*/

void shms_quad(TString basename=""){
  if (basename=="") {
    cout << " Input the basename of the root file (assumed to be in worksim)" << endl;
    cin >> basename;
  }
  gStyle->SetPalette(1,0);
  gStyle->SetOptStat(1);
  gStyle->SetOptFit(11);
  gStyle->SetTitleOffset(1.,"Y");
  gStyle->SetTitleOffset(.7,"X");
  gStyle->SetLabelSize(0.04,"XY");
  gStyle->SetTitleSize(0.06,"XY");
  gStyle->SetPadLeftMargin(0.12);
  TString inputroot;
  inputroot="worksim/"+basename+".root";
  TString outputhist;
  outputhist="newquad_scan_hist/"+basename+"_hist.root";
  TObjArray HList(0);
  TString outputpdf;
  outputpdf=basename+".pdf";
  TString htitle=basename;
  TPaveLabel *title = new TPaveLabel(.15,.90,0.95,.99,htitle,"ndc");
  //
  TFile *fsimc = new TFile(inputroot); 
  TTree *tsimc = (TTree*) fsimc->Get("h1411");
  // Define branches
  Float_t         psxfp; // position at focal plane ,+X is pointing down
  Float_t         psyfp; // X x Y = Z so +Y pointing central ray left
  Float_t         psxpfp; // dx/dz at focal plane
  Float_t         psypfp; //  dy/dz at focal plane
  Float_t         psyptar;//reconstructed
  Float_t         psyptari;//reconstructed
  Float_t         psytari;//reconstructed
  Float_t         psytar;//reconstructed
  Float_t         psxptar;//reconstructed
  Float_t         ys;//reconstructed
  Float_t         ys_calc;//reconstructed
  Float_t         xs;//reconstructed
  Float_t         xs_calc;//reconstructed
  Float_t         delta;//reconstructed
  Float_t         deltai;//reconstructed
  Float_t         evtyp;//reconstructed
  tsimc->SetBranchAddress("ysieve",&ys);
  tsimc->SetBranchAddress("xsieve",&xs);
  tsimc->SetBranchAddress("psdelta",&delta);
  tsimc->SetBranchAddress("psdeltai",&deltai);
  tsimc->SetBranchAddress("psxfp",&psxfp);
  tsimc->SetBranchAddress("psyfp",&psyfp);
  tsimc->SetBranchAddress("psxpfp",&psxpfp);
  tsimc->SetBranchAddress("psypfp",&psypfp);
  tsimc->SetBranchAddress("psytar",&psytar);
  tsimc->SetBranchAddress("psytari",&psytari);
  tsimc->SetBranchAddress("psyptar",&psyptar);
  tsimc->SetBranchAddress("psyptari",&psyptari);
  tsimc->SetBranchAddress("psxptar",&psxptar);
  tsimc->SetBranchAddress("evtype",&evtyp);
  //
  Int_t type =1;
  TH2F *hxfp_yfp = new TH2F("hxfp_yfp", Form("Event type= %d ; X_fp ; Y_fp",type), 200, -10,10,200,-20,20);
  HList.Add(hxfp_yfp);
  TH2F *hxfp_ypfp = new TH2F("hxfp_ypfp", Form("Event type= %d ; X_fp ; Yp_fp",type), 200, -10,10,200,-.1,.1);
  HList.Add(hxfp_ypfp);
  TH2F *hxfp_xpfp = new TH2F("hxfp_xpfp", Form("Event type= %d ; X_fp ; Xp_fp",type), 200, -10,10,200,-.1,.1);
  HList.Add(hxfp_xpfp);
  TH2F *hxpfp_yfp = new TH2F("hxpfp_yfp", Form("Event type = %d ; Xp_fp ; Y_fp",type), 200, -.1,.1, 200, -20,20);
  HList.Add(hxpfp_yfp);
  TH2F *hxpfp_ypfp = new TH2F("hxpfp_ypfp", Form("Event type = %d ; Xp_fp ; Yp_fp",type), 200, -.1,.1,200,-.1,.1);
  HList.Add(hxpfp_ypfp);
  TH2F *hypfp_yfp = new TH2F("hypfp_yfp", Form("Event type = %d ; Yp_fp ; Y_fp",type), 200, -.1,.1, 200, -20,20);
  HList.Add(hypfp_yfp);
  Long64_t nentries = tsimc->GetEntries();
  for (int i = 0; i < nentries; i++) {
    tsimc->GetEntry(i);
    if (evtyp == type) {
      hxfp_yfp->Fill(psxfp,psyfp);
      hxfp_ypfp->Fill(psxfp,psypfp);
      hxfp_xpfp->Fill(psxfp,psxpfp);
      hxpfp_ypfp->Fill(psxpfp,psypfp);
      hxpfp_yfp->Fill(psxpfp,psyfp);
      hypfp_yfp->Fill(psypfp,psyfp);
      
    }
  }
  TFile hsimc(outputhist,"recreate");
  HList.Write();
}   
