/*###
--- C++ interface to DiTau_ML_mass
--- https://github.com/lucastorterotot/DiTau_ML_mass
--- Davide Zuolo (University and INFN Milano - Bicocca)
--- March 2021
###*/

#include "../interface/diTauMLMassInterface.h"

namespace ditauMLMass {

diTauMLMass::diTauMLMass(const std::string & model)
{
       nn_desc.graph.reset(tensorflow::loadMetaGraph(model));
       nn_desc.session = tensorflow::createSession(nn_desc.graph.get(), model);
       nn_desc.input_layer = "serving_default_input_layer:0";
       nn_desc.output_layer = "StatefulPartitionedCall:0";
}

std::vector<float> diTauMLMass::GetScore(const float tau1_px, const float tau1_py, const float tau1_pz,const float tau1_e,const float tau2_px,const float tau2_py,const float tau2_pz,const float tau2_e,const float tau1_pt,const float tau1_eta,const float tau1_phi,const float tau2_pt,const float tau2_eta,const float tau2_phi,const float tau1_dm,const float tau2_dm,const float ditau_deltaphi,const float ditau_deltaeta,const float MET_pt, const float MET_phi,const float MET_X,const float  MET_Y,const float DeepMET_ResponseTune_px,const float DeepMET_ResponseTune_py,const float DeepMET_ResolutionTune_px,const float DeepMET_ResolutionTune_py,const float MET_covXX,const float MET_covXY,const float MET_covYY,const int PU_npvs,const float bjet1_pt, const float bjet1_eta,const float bjet1_phi,const float bjet1_deepFlavor,const float bjet1_pNet_b,const float bjet1_pNet_c ,const float bjet1_pNet_uds ,const float bjet2_pt ,const float bjet2_eta ,const float bjet2_phi ,const float bjet2_deepFlavor ,const float bjet2_pNet_b ,const float bjet2_pNet_c ,const float bjet2_pNet_uds ,const float mT1 ,const float mT2 ,const float mTtt ,const float mTtot ,const float mVis , const int N_neutrinos ,const float BDT_channel ,const float BDT_ditau_deltaPhi ,const float BDT_dib_abs_deltaPhi ,const float BDT_dib_deltaPhi ,const float BDT_dau1MET_deltaPhi ,const float BDT_bHMet_deltaPhi ,const float BDT_HT20 ,const float BDT_topPairMasses ,const float BDT_topPairMasses2 , const float BDT_bH_tauH_MET_InvMass ,const float BDT_bH_tauH_InvMass ,const float BDT_total_CalcPhi , const float BDT_ditau_CalcPhi ,const float BDT_dib_CalcPhi ,const float BDT_MET_bH_cosTheta,const float BDT_b1_bH_cosTheta )
{
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, diTauMLMass::n_variables});

    x.flat<float>().setZero();

    x.tensor<float, 2>()(0, InputVars::vars::tau1_px)          = tau1_px;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_py)          = tau1_py;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_pz)          = tau1_pz;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_e)           = tau1_e;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_px)          = tau2_px;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_py)          = tau2_py;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_pz)          = tau2_pz;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_e)           = tau2_e;
   
    x.tensor<float, 2>()(0, InputVars::vars::tau1_pt)          = tau1_pt;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_eta)         = tau1_eta;
    x.tensor<float, 2>()(0, InputVars::vars::tau1_phi)         = tau1_phi;
    
    x.tensor<float, 2>()(0, InputVars::vars::tau2_pt)          = tau2_pt;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_eta)         = tau2_eta;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_phi)         = tau2_phi;


    x.tensor<float, 2>()(0, InputVars::vars::tau1_dm)          = tau1_dm;
    x.tensor<float, 2>()(0, InputVars::vars::tau2_dm)          = tau2_dm;

    x.tensor<float, 2>()(0, InputVars::vars::ditau_deltaphi)   = ditau_deltaphi;
    x.tensor<float, 2>()(0, InputVars::vars::ditau_deltaeta)   = ditau_deltaeta;

    x.tensor<float, 2>()(0, InputVars::vars::MET_pt)           = MET_pt;
    x.tensor<float, 2>()(0, InputVars::vars::MET_phi)          = MET_phi;
    x.tensor<float, 2>()(0, InputVars::vars::MET_X)            = MET_X;
    x.tensor<float, 2>()(0, InputVars::vars::MET_Y)            = MET_Y;

    x.tensor<float, 2>()(0, InputVars::vars::DeepMET_ResponseTune_px)           = DeepMET_ResponseTune_px;
    x.tensor<float, 2>()(0, InputVars::vars::DeepMET_ResponseTune_py)           = DeepMET_ResponseTune_py;
    x.tensor<float, 2>()(0, InputVars::vars::DeepMET_ResolutionTune_px)         = DeepMET_ResolutionTune_px;
    x.tensor<float, 2>()(0, InputVars::vars::DeepMET_ResolutionTune_py)         = DeepMET_ResolutionTune_py;

    x.tensor<float, 2>()(0, InputVars::vars::MET_covXX)                         = MET_covXX;
    x.tensor<float, 2>()(0, InputVars::vars::MET_covXY)                         = MET_covXY;
    x.tensor<float, 2>()(0, InputVars::vars::MET_covYY)                         = MET_covYY;
    x.tensor<float, 2>()(0, InputVars::vars::PU_npvs)                           = PU_npvs;

    x.tensor<float, 2>()(0, InputVars::vars::bjet1_pt)                          = bjet1_pt;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_eta)                         = bjet1_eta;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_phi)                         = bjet1_phi;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_deepFlavor)                  = bjet1_deepFlavor;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_pNet_b)                      = bjet1_pNet_b;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_pNet_c)                      = bjet1_pNet_c;
    x.tensor<float, 2>()(0, InputVars::vars::bjet1_pNet_uds)                    = bjet1_pNet_uds;

    x.tensor<float, 2>()(0, InputVars::vars::bjet2_pt)                          = bjet2_pt;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_eta)                         = bjet2_eta;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_phi)                         = bjet2_phi;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_deepFlavor)                  = bjet2_deepFlavor;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_pNet_b)                      = bjet2_pNet_b;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_pNet_c)                      = bjet2_pNet_c;
    x.tensor<float, 2>()(0, InputVars::vars::bjet2_pNet_uds)                    = bjet2_pNet_uds;

    x.tensor<float, 2>()(0, InputVars::vars::mT1)                               = mT1;
    x.tensor<float, 2>()(0, InputVars::vars::mT2)                               = mT2;
    x.tensor<float, 2>()(0, InputVars::vars::mTtt)                              = mTtt;
    x.tensor<float, 2>()(0, InputVars::vars::mTtot)                             = mTtot;

    x.tensor<float, 2>()(0, InputVars::vars::mVis )                             = mVis ;
    x.tensor<float, 2>()(0, InputVars::vars::N_neutrinos)                       = N_neutrinos;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_channel)                       = BDT_channel;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_ditau_deltaPhi )               = BDT_ditau_deltaPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_dib_abs_deltaPhi)              = BDT_dib_abs_deltaPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_dib_deltaPhi )                 = BDT_dib_deltaPhi ;

    x.tensor<float, 2>()(0, InputVars::vars::BDT_dau1MET_deltaPhi)              = BDT_dau1MET_deltaPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_bHMet_deltaPhi)                = BDT_bHMet_deltaPhi ;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_HT20)                          = BDT_HT20;

    x.tensor<float, 2>()(0, InputVars::vars::BDT_topPairMasses)                 = BDT_topPairMasses;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_topPairMasses2)                = BDT_topPairMasses2;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_bH_tauH_MET_InvMass)           = BDT_bH_tauH_MET_InvMass;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_bH_tauH_InvMass)               = BDT_bH_tauH_InvMass;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_total_CalcPhi)                 = BDT_total_CalcPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_ditau_CalcPhi)                 = BDT_ditau_CalcPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_dib_CalcPhi)                   = BDT_dib_CalcPhi;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_MET_bH_cosTheta)               = BDT_MET_bH_cosTheta;
    x.tensor<float, 2>()(0, InputVars::vars::BDT_b1_bH_cosTheta)                = BDT_b1_bH_cosTheta;
    


    std::vector<tensorflow::Tensor> pred_vec; 

    std::vector<float> output; //std::vector < std::vector<float> > output;

    tensorflow::run(nn_desc.session, { { nn_desc.input_layer, x } },{ nn_desc.output_layer }, &pred_vec);


     // Get the shape of the vector
/*	tensorflow::TensorShape vector_shape;
	for (const auto& tensor : pred_vec) {
	  if (vector_shape.dims() == 0) {
	    vector_shape = tensor.shape();
	  } else {
	    for (int i = 0; i < tensor.shape().dims(); ++i) {
	      vector_shape.set_dim(i, vector_shape.dim_size(i) + tensor.shape().dim_size(i));
	    }
	  }
	}

     // Print the shape of the vector
	std::cout << "Vector shape: " << vector_shape.DebugString() << std::endl; */
	
   //nu1_px, py, pz, e
    output.push_back(pred_vec.at(0).matrix<float>()(0)); 
    output.push_back(pred_vec.at(0).matrix<float>()(1)); 
    output.push_back(pred_vec.at(0).matrix<float>()(2)); 
    output.push_back(pred_vec.at(0).matrix<float>()(3));
   //nu2_px, py, pz, e
    output.push_back(pred_vec.at(0).matrix<float>()(4)); 
    output.push_back(pred_vec.at(0).matrix<float>()(5)); 
    output.push_back(pred_vec.at(0).matrix<float>()(6)); 
    output.push_back(pred_vec.at(0).matrix<float>()(7));
   //minv2
    output.push_back(pred_vec.at(0).matrix<float>()(8));
   //classifier 
    output.push_back(pred_vec.at(0).matrix<float>()(9)); 
    output.push_back(pred_vec.at(0).matrix<float>()(10)); 
    output.push_back(pred_vec.at(0).matrix<float>()(11));

    return output;
}

diTauMLMass::~diTauMLMass()
{
    tensorflow::closeSession(nn_desc.session);
}

}// namespace ditauMLMass
