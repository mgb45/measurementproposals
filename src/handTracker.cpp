#include "handTracker.h"

using namespace cv;
using namespace std;

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/likelihood",1); //ROS
	hand_face_pub = nh.advertise<handBlobTracker::HFPose2DArray>("/faceHandPose", 10);
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
		
	tempSL = 0;
	tempSR = 0;
	
	cv::Mat subImg1 = cv::Mat::zeros(50,50,CV_8UC3);
	
	int histSize[] = {20,20};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, rangesh, true, false);
	
	pMOG2 = new BackgroundSubtractorMOG2();
	  
	double dt = 0.1; // actually 1/ 30fps, but uncertainty in acc tuned for this
	for (int i = 0; i < 2; i++)
	{
		cv::KalmanFilter KF;
		KF.init(6,2,0,CV_32F);
		KF.transitionMatrix = *(Mat_<float> (6, 6) << 1, 0, dt, 0, dt*dt, 0, 
													0, 1, 0, dt, 0, dt*dt, 
													0, 0, 1, 0, dt, 0, 
													0, 0, 0, 1, 0, dt, 
													0, 0, 0, 0, 1, 
													0, 0, 0, 0, 0, 0, 1);
		KF.statePre.at<float>(0) = 0;
		KF.statePre.at<float>(1) = 0;
		KF.statePre.at<float>(2) = 0;
		KF.statePre.at<float>(3) = 0;
		KF.statePre.at<float>(4) = 0;
		KF.statePre.at<float>(5) = 0;
		
		setIdentity(KF.measurementMatrix);
		KF.processNoiseCov = *(Mat_<float> (6, 6) << 0, 0, 0, 0, 0, 0, 
													0, 0, 0, 0, 0, 0, 
													0, 0, 0, 0, 0, 0,
													0, 0, 0, 0, 0, 0,
													0, 0, 0, 0, dt*100000, 0,
													0, 0, 0, 0, 0, dt*100000);
		setIdentity(KF.measurementNoiseCov, Scalar::all(5));
		setIdentity(KF.errorCovPost, Scalar::all(500000));
		setIdentity(KF.errorCovPre, Scalar::all(500000));
		tracked[i] = false;
		tracker.push_back(KF);
		box.push_back(cv::RotatedRect(Point2f(0,0),Size2f(0,0),0));
	}
}

HandTracker::~HandTracker()
{
	delete sync;
}

void HandTracker::checkHandsInitialisation(cv::Mat likelihood, cv::Mat image3, double xShift,cv::RotatedRect &roi, bool &track)
{
	/********Left right hand initialisation ************/
	if (!track)
	{
		cv::Rect temp;
		temp.x = image3.cols/16 + xShift;
		temp.y = image3.rows/8;
		temp.width = image3.cols/2-2*image3.cols/16;
		temp.height = image3.rows - 2*image3.rows/8;
		rectangle(image3, temp, Scalar(255,0,0), 2, 8, 0);
		
		roi = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		roi.size.height = min((int)roi.size.height,20);
		roi.size.width = min((int)roi.size.width,20);
		temp = roi.boundingRect();
		
		temp = adjustRect(temp,image3.size());
		
		try
		{
			ellipse(image3, roi, Scalar(255,0,0), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
		
		tempSR = cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
		ROS_DEBUG("Init: %f",tempSR);
		if ((tempSR < lScoreInit)||(temp.width <= 5)||(temp.height <= 5))
		{
			track = false;
		}
		else
		{
			if (xShift > 0)
			{
				tracker[1].statePre.at<float>(0) = roi.center.x;
				tracker[1].statePre.at<float>(1) = roi.center.y;
				tracker[1].statePre.at<float>(2) = 0;
				tracker[1].statePre.at<float>(3) = 0;
				tracker[1].statePre.at<float>(4) = 0;
				tracker[1].statePre.at<float>(5) = 0;
				setIdentity(tracker[1].errorCovPost, Scalar::all(500000));
				setIdentity(tracker[1].errorCovPre, Scalar::all(500000));
			}
			else
			{
				tracker[0].statePre.at<float>(0) = roi.center.x;
				tracker[0].statePre.at<float>(1) = roi.center.y;
				tracker[0].statePre.at<float>(2) = 0;
				tracker[0].statePre.at<float>(3) = 0;
				tracker[0].statePre.at<float>(4) = 0;
				tracker[0].statePre.at<float>(5) = 0;
				setIdentity(tracker[0].errorCovPost, Scalar::all(500000));
				setIdentity(tracker[0].errorCovPre, Scalar::all(500000));
			}
			track = true;
			ellipse(image3, roi, Scalar(0,255,0), 2, 8);
		}
	}
}

cv::Rect HandTracker::adjustRect(cv::Rect temp,cv::Size size)
{
	cv::Rect newRect;
	newRect.x = min(max(temp.x,1),size.width-5);
	newRect.y = min(max(temp.y,1),size.height-5);
	newRect.width = min(temp.width,size.width-newRect.x);
	newRect.height = min(temp.height,size.height-newRect.y);
	return newRect;
}

void HandTracker::updateHandPos(cv::Mat likelihood, cv::Mat image3, cv::RotatedRect &roi, bool &track, face &face_in)
{
	if (track)
	{
		roi.size.width = roi.size.width*2.1;
		roi.size.height = roi.size.height*2.1;
		cv::Rect temp = roi.boundingRect();
		
		try
		{
			ellipse(image3, roi, Scalar(0,255,255), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
		
		temp = adjustRect(temp,image3.size());
		roi = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		roi.size.height = min((int)roi.size.height,face_in.roi.height);
		roi.size.width = min((int)roi.size.width,face_in.roi.width);
		temp = roi.boundingRect();
		temp = adjustRect(temp,image3.size());
		tempSR = tempSR*0.6 + 0.4*cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
		ROS_DEBUG("Track: %f",tempSR);
		if ((tempSR < lScoreThresh)||(temp.width <= 1)||(temp.height <= 1))
		{
			ROS_WARN ("Lost Hand! %f %d %d",tempSR,temp.width,temp.height);
			track = false;
		}
					
		try
		{
			ellipse(image3, roi, Scalar(255,255,0), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
	}
}

// Hand detector: given likelihood of skin, detects hands and intialising area and uses camshift to track them
void HandTracker::HandDetector(cv::Mat likelihood, face &face_in, cv::Mat image3)
{
	//Set face probability to zero, not a hand
	cv::Rect roi_enlarged; // enlarge face to cover neck and ear blobs
	roi_enlarged.height = face_in.roi.height*1.8;
	roi_enlarged.width = face_in.roi.width*1.5;
	roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	roi_enlarged.y = face_in.roi.height/2 + face_in.roi.y - roi_enlarged.height/3;
	roi_enlarged = adjustRect(roi_enlarged,image3.size());
		
	ellipse(likelihood, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);
	
	cvtColor(likelihood,image3,CV_GRAY2RGB);
	
	std::vector<cv::Mat> tempLikelihood;
	tempLikelihood.push_back(cv::Mat(likelihood.rows,likelihood.cols,CV_8UC1));
	tempLikelihood.push_back(cv::Mat(likelihood.rows,likelihood.cols,CV_8UC1));

	for (int i = 0; i < 2; i++)
	{
		if (tracked[i])
		{
			// KF prediction
			cv::Mat prediction = tracker[i].predict();
			box[i].center.x = prediction.at<float>(0);
			box[i].center.y = prediction.at<float>(1);
			cv::Mat eigenvals, eigenvecs;
			eigen(tracker[i].errorCovPre(Range(0,2),Range(0,2)),eigenvals,eigenvecs);
			try
			{
				ellipse(image3,RotatedRect(Point2f(box[i].center.x,box[i].center.y),Size2f(3*sqrt(eigenvals.at<float>(0,0)),3*sqrt(eigenvals.at<float>(1,0))),atan2(eigenvecs.at<float>(0,0),eigenvecs.at<float>(1,0))), Scalar(255,0,0), 2, 8);
			}
			catch (cv::Exception& e )
			{
			}
		}
	}
	for (int i = 0; i < 2; i++)
	{
		likelihood.copyTo(tempLikelihood[i]);
		if (tracked[i])
		{
			// Mask opposite hand ellipse in likelihood image
			cv::Mat roiMask = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
			cv::Mat roiMask1 = cv::Mat::ones(image3.rows,image3.cols,CV_8UC1)*255;
			cv::Mat roiMask2 = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
			cv::RotatedRect abox_t = box[(int)(!i)];
			abox_t.size.width = abox_t.size.width*2;
			abox_t.size.height = abox_t.size.height*2;
			cv::RotatedRect bbox_t = box[i];
			bbox_t.size.width = bbox_t.size.width*1.0;
			bbox_t.size.height = bbox_t.size.height*1.0;
			try
			{
				ellipse(roiMask1, abox_t, Scalar(0,0,0), -1, 8);
				ellipse(roiMask2, bbox_t, Scalar(255,255,255), -1, 8);
			}
			catch (cv::Exception& e)
			{
				const char* err_msg = e.what();
				ROS_ERROR("%s",err_msg);
			}	
		
			bitwise_or(roiMask1,roiMask2,roiMask); // R' + L => Blocks R
			bitwise_and(roiMask,likelihood,tempLikelihood[i]);
		}
	}
		
	for (int i = 0; i < 2; i++)
	{
		checkHandsInitialisation(tempLikelihood[i],image3,(double)i*(double)likelihood.cols/2.0,box[i],tracked[i]);
		updateHandPos(tempLikelihood[i], image3, box[i], tracked[i], face_in);
		Mat_<float> measurement(2,1);
		if (tracked[i])
		{
			measurement.at<float>(0) = box[i].center.x;
			measurement.at<float>(1) = box[i].center.y;
			cv::Mat estimated = tracker[i].correct(measurement);
			box[i].center.x = estimated.at<float>(0);
			box[i].center.y = estimated.at<float>(1);
		}
	}
	
	// Sanity check on same hand measurements
	if ((box[0].center.x == box[1].center.x)&&(box[0].center.y == box[1].center.y))
	{
		tracked[0]  = false;
		tracked[1] = false;
	}
		
	ROS_DEBUG("Exit: %d %d",tracked[0],tracked[1]);
}

// Gets skin colour likelihood map from face using back projection in Lab
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image4;
	cvtColor(input,image4,CV_BGR2Lab);
				
	MatND hist;
	int histSize[] = {20,20};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	
	cv::Rect rec_reduced;
	rec_reduced.x = face_in.roi.x+ face_in.roi.width/4;
	rec_reduced.y = face_in.roi.y+ face_in.roi.height/4;
	rec_reduced.width = face_in.roi.width - 2*face_in.roi.width/4;
	rec_reduced.height = face_in.roi.height- 2*face_in.roi.height/4;
	
	pMOG2->operator()(input,fgMaskMOG2,-10);
	
	// Generate output image
	cv::Mat foreground(image4.size(),CV_8UC3,cv::Scalar(255,255,255)); // all white image
	image4.copyTo(foreground,fgMaskMOG2); // bg pixels not copied
	
	cv::Mat subImg1 = image4(rec_reduced);
	
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist,2,histSize, rangesh, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	hist1 = 0.95*hist1 + 0.05*hist;
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	calcBackProject(&foreground,1,channels,hist1,temp1,rangesh, 1, true);
	
	Mat element0 = getStructuringElement(MORPH_ELLIPSE, Size(7,7), Point(3,3));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(11,11), Point(5,5));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	
	dilate(temp1,temp1,element0);
	erode(temp1,temp1,element1);
	dilate(temp1,temp1,element2);

	cv::Mat image3 = cv::Mat::zeros(image4.rows,image4.cols,CV_8UC3);
	
	ROS_DEBUG("Entry: %d %d",tracked[0],tracked[1]);

	// Detect hands and update pose estimate
	HandDetector(temp1,face_in,image3);
				
	// Draw face rectangles on display image
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(rec_reduced.x,rec_reduced.y), Point(rec_reduced.x+rec_reduced.width,rec_reduced.y+rec_reduced.height), Scalar(0,255,0), 4, 8, 0);
		
	return image3;
}

// Update tracked face with latest info
void HandTracker::updateFaceInfo(const faceTracking::ROIArrayConstPtr& msg)
{
	if (msg->ROIs.size() > 0)
	{
		if (face_found.views <= 0) // if no faces exist yet
		{
			//pick first face in list
			face_found.roi = cv::Rect(msg->ROIs[0].x_offset,msg->ROIs[0].y_offset,msg->ROIs[0].width,msg->ROIs[0].height);
			face_found.id = msg->ids[0];
			face_found.views = 1;
			tracked[0] = false;
			tracked[1] = false;
		}
		else
		{
			face_found.views--;
			for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list, iterate through detected faces
			{
				if (face_found.id.compare(msg->ids[i]) == 0) // old face
				{
					face_found.roi = cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height); //update roi
					face_found.id = msg->ids[i];
					face_found.views++;
				}
			}
		}
	}
	else
	{
		face_found.views = 0;
	}
}

// image and face roi callback
void HandTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg)
{
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
			
		updateFaceInfo(msg); // update face list
		
		cv::Mat outputImage = cv::Mat::zeros(image.rows,image.cols,image.type()); // display image
		if (face_found.views > 0) // get hands
		{
			outputImage = getHandLikelihood(image,face_found);
			tracked[0] = tracked[0]&tracked[1];
			tracked[1] = tracked[0];
		}
		else
		{
			tracked[0] = false;
			tracked[1] = false;
		}
		cv_bridge::CvImage img2;
		img2.encoding = "rgb8";
		img2.header = immsg->header;
		img2.image = outputImage;			
		pub.publish(img2.toImageMsg()); // publish result image
		
		handBlobTracker::HFPose2D rosHands;
		handBlobTracker::HFPose2DArray rosHandsArr;
		for (int i = 0; i < 2; i++)
		{
			rosHands.x = box[i].center.x;
			rosHands.y = box[i].center.y;
			rosHandsArr.measurements.push_back(rosHands);
			rosHandsArr.valid.push_back(tracked[i]);
		}
		rosHandsArr.names.push_back("Left Hand");
		rosHandsArr.names.push_back("Right Hand");
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + int(face_found.roi.height/2.0);
		rosHandsArr.measurements.push_back(rosHands);
		rosHandsArr.names.push_back("Head");
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + 3.25/2.0*face_found.roi.height;
		rosHandsArr.measurements.push_back(rosHands); //Neck
		rosHandsArr.names.push_back("Neck");
		rosHandsArr.valid.push_back(true);
		rosHandsArr.valid.push_back(true);
		rosHandsArr.header = msg->header;
		rosHandsArr.id = face_found.id;
		hand_face_pub.publish(rosHandsArr);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}
