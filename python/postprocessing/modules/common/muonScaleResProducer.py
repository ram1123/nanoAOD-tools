from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import ROOT
import os
import random
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import MuonScaRe
import numpy as np
from math import pi

def mk_safe(fct, *args):
    try:
        return fct(*args)
    except Exception as e:
        if any('Error in function boost::math::erf_inv' in arg
               for arg in e.args):
            print(
                'WARNING: catching exception and returning -1. Exception arguments: %s'
                % e.args)
            return -1.
        else:
            raise e


class muonScaleResProducer(Module):
    def __init__(self, rc_dir, rc_corrections, dataYear, dataEra):
        self.dataEra = dataEra
        if self.dataEra == "Run2":
            p_postproc = '%s/src/PhysicsTools/NanoAODTools/python/postprocessing' % os.environ[
                'CMSSW_BASE']
            p_roccor = p_postproc + '/data/' + rc_dir
            if "/RoccoR_cc.so" not in ROOT.gSystem.GetLibraries():
                p_helper = '%s/RoccoR.cc' % p_roccor
                print('Loading C++ helper from ' + p_helper)
                ROOT.gROOT.ProcessLine('.L ' + p_helper)
            self._roccor = ROOT.RoccoR(p_roccor + '/' + rc_corrections)
        elif self.dataEra == "Run3":
            json = "%s/src/PhysicsTools/NanoAODTools/python/postprocessing/data/%s/%s" % (os.environ['CMSSW_BASE'],rc_dir,  rc_corrections)
            print (json)
            self.corrModule = MuonScaRe(json)

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        if self.dataEra == "Run2":
            self.out.branch("Muon_corrected_pt", "F", lenVar="nMuon")
            self.out.branch("Muon_correctedUp_pt", "F", lenVar="nMuon")
            self.out.branch("Muon_correctedDown_pt", "F", lenVar="nMuon")
        if self.dataEra == "Run3":
            self.out.branch("Muon_pt", "F", lenVar="nMuon")
            self.out.branch("Muon_uncorrected_pt", "F", lenVar="nMuon")
            self.out.branch("Muon_syst_pt", "F", lenVar="nMuon")
            self.out.branch("Muon_stat_pt", "F", lenVar="nMuon")
        self.is_mc = bool(inputTree.GetBranch("GenJet_pt"))

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def getPtCorr(self, muon, var = "nom") :

        if muon.pt < 26 or muon.pt > 200:
            return muon.pt

        isData = int(not self.is_mc)

        scale_corr = self.corrModule.pt_scale(isData, muon.pt, muon.eta, muon.phi, muon.charge, var)
        pt_corr = scale_corr

        if self.is_mc:
            smear_corr = self.corrModule.pt_resol(scale_corr, muon.eta, muon.nTrackerLayers, var)
            pt_corr = smear_corr

        return pt_corr

    def analyze(self, event):
        muons = Collection(event, "Muon")
        if self.dataEra == "Run2":
            if self.is_mc:
                genparticles = Collection(event, "GenPart")
            roccor = self._roccor
            if self.is_mc:
                pt_corr = []
                pt_err = []
                for mu in muons:
                    genIdx = mu.genPartIdx
                    if genIdx >= 0 and genIdx < len(genparticles):
                        genMu = genparticles[genIdx]
                        pt_corr.append(mu.pt *
                                    mk_safe(roccor.kSpreadMC, mu.charge, mu.pt,
                                            mu.eta, mu.phi, genMu.pt))
                        pt_err.append(mu.pt *
                                    mk_safe(roccor.kSpreadMCerror, mu.charge,
                                            mu.pt, mu.eta, mu.phi, genMu.pt))
                    else:
                        u1 = random.uniform(0.0, 1.0)
                        pt_corr.append(
                            mu.pt * mk_safe(roccor.kSmearMC, mu.charge, mu.pt,
                                            mu.eta, mu.phi, mu.nTrackerLayers, u1))
                        pt_err.append(
                            mu.pt * mk_safe(roccor.kSmearMCerror, mu.charge, mu.pt,
                                            mu.eta, mu.phi, mu.nTrackerLayers, u1))

            else:
                pt_corr = list(
                    mu.pt *
                    mk_safe(roccor.kScaleDT, mu.charge, mu.pt, mu.eta, mu.phi)
                    for mu in muons)
                pt_err = list(
                    mu.pt *
                    mk_safe(roccor.kScaleDTerror, mu.charge, mu.pt, mu.eta, mu.phi)
                    for mu in muons)

            self.out.fillBranch("Muon_corrected_pt", pt_corr)
            pt_corr_up = list(
                max(pt_corr[imu] + pt_err[imu], 0.0)
                for imu, mu in enumerate(muons))
            pt_corr_down = list(
                max(pt_corr[imu] - pt_err[imu], 0.0)
                for imu, mu in enumerate(muons))
            self.out.fillBranch("Muon_correctedUp_pt", pt_corr_up)
            self.out.fillBranch("Muon_correctedDown_pt", pt_corr_down)

        if self.dataEra == "Run3":
            pt_corr = [0.]*len(muons)
            pt_syst = [0.]*len(muons)
            pt_stat = [0.]*len(muons)

            isData = int(not self.is_mc)

            for imu, muon in enumerate(muons):
                seedSeq = np.random.SeedSequence([event.luminosityBlock, event.event, int(abs((muon.phi/pi*100.)%1)*1e10), 351740215])
                self.corrModule.setSeed(int(seedSeq.generate_state(1,np.uint64)[0]))

                pt_corr[imu] = self.getPtCorr(muon, "nom")
                pt_syst[imu] = self.getPtCorr(muon, "syst")
                pt_stat[imu] = self.getPtCorr(muon, "stat")

                pt_uncorr = list(mu.pt for mu in muons)
                self.out.fillBranch("Muon_uncorrected_pt", pt_uncorr)
                self.out.fillBranch("Muon_pt", pt_corr)
                self.out.fillBranch("Muon_syst_pt", pt_syst)
                self.out.fillBranch("Muon_stat_pt", pt_stat)

        return True

# Ref: https://twiki.cern.ch/twiki/bin/view/CMS/RochcorMuon & https://muon-wiki.docs.cern.ch/guidelines/corrections/?h=roch#medium-pt-trigger-efficiencies
muonScaleRes2016pre = lambda: muonScaleResProducer('roccor.Run2.v5',
                                                'RoccoR2016aUL.txt', 2016, "Run2")
muonScaleRes2016 = lambda: muonScaleResProducer('roccor.Run2.v5',
                                                'RoccoR2016bUL.txt', 2016, "Run2")
muonScaleRes2017 = lambda: muonScaleResProducer('roccor.Run2.v5',
                                                'RoccoR2017UL.txt', 2017, "Run2")
muonScaleRes2018 = lambda: muonScaleResProducer('roccor.Run2.v5',
                                                'RoccoR2018UL.txt', 2018, "Run2")
muonScaleRes2022 = lambda: muonScaleResProducer('roccor.2022.v2',
                                                '2022_schemaV2.json', 2022, "Run3")
muonScaleRes2022EE = lambda: muonScaleResProducer('roccor.2022.v2',
                                                '2022EE_schemaV2.json', '2022EE', "Run3")
muonScaleRes2023 = lambda: muonScaleResProducer('roccor.2023.v2',
                                                '2023_schemaV2.json', 2023, "Run3")
muonScaleRes2023BPix = lambda: muonScaleResProducer('roccor.2023.v2',
                                                '2023BPix_schemaV2.json', '2023BPix', "Run3")
