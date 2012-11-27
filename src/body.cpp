#include "body.h"

using namespace std;
using namespace cv;

body::body()
{
}

body::body(cv::Rect roi_in, int N)
{
	roi = roi_in;
	model_params temp;
	seen = true;
	for (int k = 0; k < N; k++)
	{
		temp.mean = k*255/(double)N;
		temp.sigma = 15;
		temp.weight = 1.0/(double)N;
		gmm_params1.push_back(temp);
		gmm_params2.push_back(temp);
	}
	pf1 = new ParticleFilter(5000,8,0);
	pf2 = new ParticleFilter(5000,8,1);
	
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
    
    rightHand.z = -1;
    leftHand.z = -1;
    
	for (int i = 0; i < 8; i++)
	{
		pf2->gmm.loadGaussian(means1.row(i),covs1(Range(8*i,8*(i+1)),Range(0,8)),weights1.at<double>(0,i));
		pf1->gmm.loadGaussian(means2.row(i),covs2(Range(8*i,8*(i+1)),Range(0,8)),weights2.at<double>(0,i));
	}
}

body::body(const body& other)
{
	id = other.id;
	roi = other.roi;
	seen = other.seen;
	gmm_params1 = other.gmm_params1;
	gmm_params2 = other.gmm_params2;
	leftHand = other.leftHand;
	rightHand = other.rightHand;
	pf1 = new ParticleFilter(5000,8,0);
	pf2 = new ParticleFilter(5000,8,1);
	*pf1 = *other.pf1;
	*pf2 = *other.pf2;
	/*pf1->N = other.pf1->N;
	pf1->d = other.pf1->d;
	pf1->particles = other.pf2->particles;
	pf1->weights = other.pf2->weights;
	pf1->im_height = other.pf2->im_height;
	pf1->im_width = other.pf2->im_width;
	pf1->gmm = other.pf1->gmm;
	pf2->N = other.pf2->N;
	pf2->d = other.pf2->d;
	pf2->particles = other.pf2->particles;
	pf2->weights = other.pf2->weights;
	pf2->im_height = other.pf2->im_height;
	pf2->im_width = other.pf2->im_width;
	pf2->gmm = other.pf2->gmm;*/
}

body body::operator=( const body& other)
{
	body newBody;
	newBody.id = other.id;
	newBody.roi = other.roi;
	newBody.seen = other.seen;
	newBody.gmm_params1 = other.gmm_params1;
	newBody.gmm_params2 = other.gmm_params2;
	newBody.leftHand = other.leftHand;
	newBody.rightHand = other.rightHand;
	newBody.pf1 = new ParticleFilter(5000,8,0);
	newBody.pf2 = new ParticleFilter(5000,8,1);
	*newBody.pf1 = *other.pf1;
	*newBody.pf2 = *other.pf2;
	return newBody;
}

body::~body()
{
	gmm_params1.clear();
	gmm_params2.clear();
	delete pf1;
	delete pf2;
}

