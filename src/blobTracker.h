#ifndef __BLOBTRACKER
#define __BLOBTRACKER

#include "body.h"
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
#include "opencv2/ml/ml.hpp"


class BlobTracker
{
	public:
		BlobTracker();
		~BlobTracker();
		cv::Mat getHistogram(cv::Mat input, cv::Rect roi, std::vector<model_params> gmm_params1, std::vector<model_params> gmm_params2, std::vector<model_params> gmm_params3);
		
	private:
		ros::Publisher pos_estimate;
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		
		void callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg); // Detected face array/ image callback
		GMM *gmm_model; // GMM used for skin colour learning
	//	cv::EM *gmm1;
		bool im_received; 
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, faceTracking::ROIArray> MySyncPolicy; // synchronising image and face array
		message_filters::Synchronizer<MySyncPolicy>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<faceTracking::ROIArray> roi_sub;
		std::vector<body> faces;
		
		void updateBlobFaces(const faceTracking::ROIArrayConstPtr& msg); 
		cv::Mat segmentFaces(cv::Mat input, body &body_in); 
		cv::SimpleBlobDetector::Params *params;
		void blobDetector(cv::Mat output, body &body_in, cv::Mat image3);
		double blobDist(geometry_msgs::Point point, cv::KeyPoint keypoint);
		
	//	cv::Mat likelihoods1, labels1, probs1;
};

#endif

