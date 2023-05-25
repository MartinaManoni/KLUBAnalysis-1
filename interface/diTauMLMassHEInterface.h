/*###
--- C++ interface to DiTau_ML_mass
--- https://github.com/lucastorterotot/DiTau_ML_mass
--- Davide Zuolo (University and INFN Milano - Bicocca)
--- March 2021
###*/

#include <vector>
#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

namespace ditauMLMassHighEven{

namespace InputVars{
    enum vars {tau1_px = 0, tau1_py = 1, tau1_pz = 2, tau1_e = 3, tau2_px = 4, tau2_py = 5, tau2_pz = 6, tau2_e = 7, tau1_pt = 8, tau1_eta = 9, tau1_phi = 10, tau2_pt = 11, tau2_eta = 12, tau2_phi = 13, tau1_dm = 14, tau2_dm = 15, ditau_deltaphi =16, ditau_deltaeta =17, MET_pt=18, MET_phi=19, MET_X = 20, MET_Y = 21, DeepMET_ResponseTune_px = 22, DeepMET_ResponseTune_py =23, DeepMET_ResolutionTune_px = 24, DeepMET_ResolutionTune_py = 25, MET_covXX = 26, MET_covXY = 27, MET_covYY = 28, PU_npvs = 29, bjet1_pt
 = 30, bjet1_eta = 31, bjet1_phi = 32, bjet1_deepFlavor =33, bjet1_pNet_b =34, bjet1_pNet_c =35, bjet1_pNet_uds =36, bjet2_pt = 37, bjet2_eta =38, bjet2_phi =39, bjet2_deepFlavor =40, bjet2_pNet_b =41, bjet2_pNet_c = 42, bjet2_pNet_uds = 43, mT1 = 44, mT2 = 45, mTtt = 46, mTtot = 47, mVis = 48, N_neutrinos =49, BDT_channel = 50, BDT_ditau_deltaPhi = 51, BDT_dib_abs_deltaPhi = 52, BDT_dib_deltaPhi = 53, BDT_dau1MET_deltaPhi = 54, BDT_bHMet_deltaPhi = 55, BDT_HT20 =56, BDT_topPairMasses =57, BDT_topPairMasses2 =58, BDT_bH_tauH_MET_InvMass = 59, BDT_bH_tauH_InvMass = 60, BDT_total_CalcPhi = 61, BDT_ditau_CalcPhi =62, BDT_dib_CalcPhi = 63, BDT_MET_bH_cosTheta = 64, BDT_b1_bH_cosTheta = 65
    };
}

class diTauMLMassHighEven {
public:
    static constexpr size_t n_variables = 66;

    diTauMLMassHighEven(const std::string& model);
    ~diTauMLMassHighEven();

    struct NNDescriptor {
        std::unique_ptr<tensorflow::MetaGraphDef> graph;
        tensorflow::Session* session;
        std::string input_layer;
        std::string output_layer;
    };

    std::vector <float> GetScore(const float tau1_px, const float tau1_py, const float tau1_pz,const float tau1_e,const float tau2_px,const float tau2_py,const float tau2_pz,const float tau2_e,const float tau1_pt,const float tau1_eta,const float tau1_phi,const float tau2_pt,const float tau2_eta,const float tau2_phi,const float tau1_dm,const float tau2_dm,const float ditau_deltaphi,const float ditau_deltaeta,const float MET_pt, const float MET_phi,const float MET_X,const float MET_Y,const float DeepMET_ResponseTune_px,const float DeepMET_ResponseTune_py,const float DeepMET_ResolutionTune_px,const float DeepMET_ResolutionTune_py,const float MET_covXX,const float MET_covXY,const float MET_covYY,const int PU_npvs,const float bjet1_pt, const float bjet1_eta,const float bjet1_phi,const float bjet1_deepFlavor,const float bjet1_pNet_b,const float bjet1_pNet_c ,const float bjet1_pNet_uds ,const float bjet2_pt ,const float bjet2_eta ,const float bjet2_phi ,const float bjet2_deepFlavor ,const float bjet2_pNet_b ,const float bjet2_pNet_c ,const float bjet2_pNet_uds ,const float mT1 ,const float mT2 ,const float mTtt ,const float mTtot ,const float mVis , const int N_neutrinos ,const float BDT_channel ,const float BDT_ditau_deltaPhi ,const float BDT_dib_abs_deltaPhi ,const float BDT_dib_deltaPhi ,const float BDT_dau1MET_deltaPhi ,const float BDT_bHMet_deltaPhi ,const float BDT_HT20 ,const float BDT_topPairMasses ,const float BDT_topPairMasses2 , const float BDT_bH_tauH_MET_InvMass ,const float BDT_bH_tauH_InvMass ,const float BDT_total_CalcPhi , const float BDT_ditau_CalcPhi ,const float BDT_dib_CalcPhi ,const float BDT_MET_bH_cosTheta,const float BDT_b1_bH_cosTheta
                   );

private:
    NNDescriptor nn_desc;
};
}// namespace ditauMLMass
