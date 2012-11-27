#include "gmm.h"

using namespace cv;
using namespace std;

GMM::GMM(int maxScale)
{
	M = maxScale;
}

GMM::~GMM()
{
}

double * GMM::expectation(std::vector<model_params> params, double x)
{
	double sum = 0;
	double *y = new double[(int)params.size()];
	for (int i = 0; i < (int)params.size(); i++)
	{
		y[i] = params[i].weight*1.0/(params[i].sigma*sqrt(2*M_PI))*exp(-pow(std::min(x-params[i].mean,M-(x-params[i].mean)),2)/(2*pow(params[i].sigma,2)));
		sum = sum + y[i];
	}
	
	for (int i = 0; i < (int)params.size(); i++)
	{
		y[i] = y[i]/sum;
	}
	return y;
}

std::vector<model_params> GMM::expectationMaximisation(std::vector<model_params> params,const cv::Mat obs)
{
	std::vector<model_params> new_params;
	new_params.resize((int)params.size());
	
	double sum[(int)params.size()];
	double *lk[obs.rows*obs.cols];
	//initialise
	
	for (int k = 0; k < (int)params.size(); k++)
	{
		new_params[k].mean = 0;
		new_params[k].sigma = 0;
		sum[k] = 0;
	}

   //find uk and Nk
	for (int i = 0; i < obs.rows*obs.cols; i++)
	{
		lk[i] = (expectation(params,obs.at<uchar>(i)));
		for (int k = 0; k < (int)params.size(); k++)
		{
			if (fabs(params[k].mean-obs.at<uchar>(i)) < M - fabs(params[k].mean-obs.at<uchar>(i)))
			{
				new_params[k].mean = new_params[k].mean + lk[i][k]*obs.at<uchar>(i);
			}
			else
			{
				new_params[k].mean = new_params[k].mean + lk[i][k]*(M-obs.at<uchar>(i));
			}
			sum[k] = sum[k]+lk[i][k];
		}
	}
	for (int k = 0; k < (int)params.size(); k++)
	{
		int b = M;
		int result = static_cast<int>((new_params[k].mean/sum[k])/b);
		new_params[k].mean = new_params[k].mean/sum[k] - static_cast<double>(result) * b;
		//new_params[k].mean = std::min(new_params[k].mean,180.0);
		//new_params[k].mean = std::max(new_params[k].mean,0.0);
	}
		
	// sigmak and wk
	for (int i = 0; i < obs.rows*obs.cols; i++)
	{
		for (int k = 0; k < (int)params.size(); k++)
		{
			new_params[k].sigma = new_params[k].sigma + lk[i][k]*pow(std::min(obs.at<uchar>(i)-new_params[k].mean,M-(obs.at<uchar>(i)-new_params[k].mean)),2);
		}
		delete lk[i];
	}
	//delete lk;
	double new_sum = 0;
	for (int k = 0; k < (int)params.size(); k++)
	{
		//new_params[k].sigma = std::min(sqrt(new_params[k].sigma/sum[k]),180 - sqrt(new_params[k].sigma/sum[k]));
		new_params[k].sigma = std::max(sqrt(new_params[k].sigma/sum[k]),0.0000);
		new_params[k].weight = std::max(sum[k]/((double)obs.rows*obs.cols),0.000);
		new_sum = new_sum + new_params[k].weight;

		if ((new_params[k].sigma <= 0.005)||(new_params[k].weight <= 0.005))
		{
			new_params[k].sigma = 255;
			new_params[k].mean = rand()%M;
		}
	}
	
/*	for (int k = 0; k < (int)params.size(); k++)
	{
		new_params[k].weight = new_params[k].weight/new_sum;
		ROS_INFO ("%f %f %f", new_params[k].sigma, new_params[k].weight, new_params[k].mean);
	}*/
	return new_params;
}
