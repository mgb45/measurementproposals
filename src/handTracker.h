#ifndef __HANDTRACKER
#define __HANDTRACKER

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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sstream>
#include <string>
#include <ros/package.h>
#include "handBlobTracker/HFPose2D.h"
#include "handBlobTracker/HFPose2DArray.h"
		

#define lScoreThresh 0.05
#define lScoreInit 0.2
		
struct face {
	cv::Rect roi;
	std::string id;
	int views;
};

class HandTracker
{
	public:
		HandTracker();
		~HandTracker();
				
	private:
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		ros::Publisher hand_face_pub;
		ros::Subscriber pose_sub;
		
		void callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg); // Detected face array/ image callback
		void poseCallback(const handBlobTracker::HFPose2DArrayConstPtr& msg);
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, faceTracking::ROIArray> MySyncPolicy; // synchronising image and face array
		message_filters::Synchronizer<MySyncPolicy>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<faceTracking::ROIArray> roi_sub;
		face face_found;
		
		cv::RotatedRect lbox, rbox;
		bool rtracked, ltracked;
		cv::MatND hist1;
		
		double tempSR, tempSL;
		
		void checkHandsInitialisation(cv::Mat likelihood, cv::Mat image3, double xShift,cv::RotatedRect &box, bool &tracked);
		void updateHandPos(cv::Mat likelihood, cv::Mat image3, cv::RotatedRect &box, bool &tracked, face &face_in);
		
		void updateFaceInfo(const faceTracking::ROIArrayConstPtr& msg);
		cv::Mat getHandLikelihood(cv::Mat input, face &face_in);
		void HandDetector(cv::Mat likelihood, face &face_in, cv::Mat image3);
			
};

#endif

