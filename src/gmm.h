#ifndef __GMM
#define __GMM

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Image.h"

struct  model_params {
	double mean;
	double sigma;
	double weight;
};

class GMM
{
	public:
		GMM(int maxScale);
		~GMM();
		std::vector<model_params> expectationMaximisation(std::vector<model_params> params, cv::Mat obs);
		
	private:
		int M;
		double * expectation(std::vector<model_params> params, double x);
};


#endif

