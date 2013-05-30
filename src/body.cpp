/*
 * body.cpp
 * 
 * Class for body parameters and pose estimate pf's 
 * Copyright 2013 Michael Burke <mgb45@cam.ac.uk>
*/
#include "body.h"

using namespace std;
using namespace cv;

body::body()
{
}

//Constructor
body::body(cv::Rect roi_in, int N)
{
	roi = roi_in; // roi for face
	model_params temp;
	seen = true; // face visible
	for (int k = 0; k < N; k++) // skin colour model initial seed
	{
		temp.mean = k*255/(double)N;
		temp.sigma = 8;
		temp.weight = 1.0/(double)N;
		gmm_params1.push_back(temp);
		gmm_params2.push_back(temp);
		gmm_params3.push_back(temp);
	}
	N = 2500;
	pf1 = new ParticleFilter(N,8,0); // left arm pf
	pf2 = new ParticleFilter(N,8,1); // right arm pf
	
	// Load Kinect GMM priors
	std::stringstream ss1;
	ss1 << ros::package::getPath("handBlobTracker") << "/data1.yml";
	std::stringstream ss2;
	ss2 << ros::package::getPath("handBlobTracker") << "/data2.yml";
	cv::FileStorage fs1(ss1.str(), FileStorage::READ);
	cv::FileStorage fs2(ss2.str(), FileStorage::READ);
    cv::Mat means1, weights1;
    cv::Mat means2, weights2;
    cv::Mat covs1;
    cv::Mat covs2;
    fs1["means"] >> means1;
    fs2["means"] >> means2;
  	fs1["covs"] >> covs1;
	fs2["covs"] >> covs2;
	fs1["weights"] >> weights1;
    fs2["weights"] >> weights2;
    fs1.release();
    fs2.release();
    
    rightHand.z = -1; // hand not yet detected
    leftHand.z = -1; // hand not yet detected
    
	for (int i = 0; i < 8; i++)
	{
		pf2->gmm.loadGaussian(means1.row(i),covs1(Range(8*i,8*(i+1)),Range(0,8)),weights1.at<double>(0,i));
		pf1->gmm.loadGaussian(means2.row(i),covs2(Range(8*i,8*(i+1)),Range(0,8)),weights2.at<double>(0,i));
	}
	num_views = 1;
}

// copy constr
body::body(const body& other)
{
	id = other.id;
	roi = other.roi;
	seen = other.seen;
	gmm_params1 = other.gmm_params1;
	gmm_params2 = other.gmm_params2;
	gmm_params3 = other.gmm_params3;
	leftHand = other.leftHand;
	rightHand = other.rightHand;
	N = 2500;
	pf1 = new ParticleFilter(N,8,0);
	pf2 = new ParticleFilter(N,8,1);
	*pf1 = *other.pf1;
	*pf2 = *other.pf2;
	num_views = other.num_views;
}

// copy constr
body body::operator=( const body& other)
{
	body newBody;
	newBody.id = other.id;
	newBody.roi = other.roi;
	newBody.seen = other.seen;
	newBody.gmm_params1 = other.gmm_params1;
	newBody.gmm_params2 = other.gmm_params2;
	newBody.gmm_params3 = other.gmm_params3;
	newBody.leftHand = other.leftHand;
	newBody.rightHand = other.rightHand;
	N = 2500;
	newBody.pf1 = new ParticleFilter(N,8,0);
	newBody.pf2 = new ParticleFilter(N,8,1);
	*newBody.pf1 = *other.pf1;
	*newBody.pf2 = *other.pf2;
	num_views = other.num_views;
	return newBody;
}

body::~body()
{
	gmm_params1.clear();
	gmm_params2.clear();
	gmm_params3.clear();
	delete pf1;
	delete pf2;
}


