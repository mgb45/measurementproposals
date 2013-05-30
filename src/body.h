#ifndef __BODY
#define __BODY

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/RegionOfInterest.h"
#include "geometry_msgs/Point.h"
#include "faceTracking/ROIArray.h"
#include "gmm.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "pf2D.h"
#include <sstream>
#include <string>
#include <ros/package.h>




class body 
{
	public:
		body();
		body(cv::Rect roi_in, int N);
		body(const body& other);
		body operator=(const body& other);
		~body();
		std::string id;
		cv::Rect roi;
		bool seen;
		std::vector<model_params> gmm_params1;
		std::vector<model_params> gmm_params2;
		std::vector<model_params> gmm_params3;
		geometry_msgs::Point leftHand;
		geometry_msgs::Point rightHand;
		ParticleFilter *pf1;
		ParticleFilter *pf2;
		int num_views;
	private:
		int N;
};

#endif
