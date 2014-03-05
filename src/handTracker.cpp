#include "handTracker.h"

using namespace cv;
using namespace std;

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/handImage",1); //ROS
	hand_face_pub = nh.advertise<handBlobTracker::HFPose2DArray>("/faceHandPose", 10);
	likelihood_pub = it.advertise("/likelihood", 1);
	
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	pose_sub = nh.subscribe("/correctedFaceHandPose", 1,&HandTracker::poseCallback,this); // requires face array input
	
	//sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, roi_sub);
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
	rtracked = false;
	ltracked = false;
	
	tempSL = 0;
	tempSR = 0;
	
	cv::Mat subImg1 = cv::Mat::zeros(50,50,CV_8UC3);
	
	int histSize[] = {60,60};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, rangesh, true, false);
	
	  pMOG2 = new BackgroundSubtractorMOG2();
	  
	  double dt = 0.1;
	ltracker.init(6,2,0,CV_32F);
	ltracker.transitionMatrix = *(Mat_<float> (6, 6) << 1, 0, dt, 0, dt*dt, 0, 0, 1, 0, dt, 0, dt*dt, 0, 0, 1, 0, dt, 0, 0, 0, 0, 1, 0, dt, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
	ltracker.statePre.at<float>(0) = 0;
	ltracker.statePre.at<float>(1) = 0;
	ltracker.statePre.at<float>(2) = 0;
	ltracker.statePre.at<float>(3) = 0;
	ltracker.statePre.at<float>(4) = 0;
	ltracker.statePre.at<float>(5) = 0;
	
	setIdentity(ltracker.measurementMatrix);
	//setIdentity(ltracker.processNoiseCov, Scalar::all(1500));
	//ltracker.processNoiseCov = *(Mat_<float> (6, 6) << pow(dt,5)/20.0, 0, pow(dt,4)/8.0, 0, pow(dt,3)/6.0, 0, 
														//0, pow(dt,5)/20.0, 0, pow(dt,4)/8.0, 0, pow(dt,3)/6.0, 
														//pow(dt,4)/8.0, 0, pow(dt,3)/3.0, 0, pow(dt,2)/2.0, 0,
														//0, pow(dt,4)/8.0, 0, pow(dt,3)/3.0, 0, pow(dt,2)/2.0,
														//pow(dt,3)/6.0, 0, pow(dt,2)/2.0, 0, dt*10000, 0,
														//0, pow(dt,3)/6.0, 0, pow(dt,2)/2.0, 0, dt*10000);
	ltracker.processNoiseCov = *(Mat_<float> (6, 6) << 0, 0, 0, 0, 0, 0, 
														0, 0, 0, 0, 0, 0, 
														0, 0, 0, 0, 0, 0,
														0, 0, 0, 0, 0, 0,
														0, 0, 0, 0, dt*100000, 0,
														0, 0, 0, 0, 0, dt*100000);
	ltracker.processNoiseCov = ltracker.processNoiseCov;
	setIdentity(ltracker.measurementNoiseCov, Scalar::all(5));
	setIdentity(ltracker.errorCovPost, Scalar::all(500000));
	setIdentity(ltracker.errorCovPre, Scalar::all(500000));
	
	rtracker.init(6,2,0,CV_32F);
	rtracker.transitionMatrix = *(Mat_<float> (6, 6) << 1, 0, dt, 0, dt*dt, 0, 0, 1, 0, dt, 0, dt*dt, 0, 0, 1, 0, dt, 0, 0, 0, 0, 1, 0, dt, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
	rtracker.statePre.at<float>(0) = 0;
	rtracker.statePre.at<float>(1) = 0;
	rtracker.statePre.at<float>(2) = 0;
	rtracker.statePre.at<float>(3) = 0;
	rtracker.statePre.at<float>(4) = 0;
	rtracker.statePre.at<float>(5) = 0;
	
	setIdentity(rtracker.measurementMatrix);
	//setIdentity(rtracker.processNoiseCov, Scalar::all(1500));
	//rtracker.processNoiseCov = *(Mat_<float> (6, 6) << pow(dt,5)/20.0, 0, pow(dt,4)/8.0, 0, pow(dt,3)/6.0, 0, 
														//0, pow(dt,5)/20.0, 0, pow(dt,4)/8.0, 0, pow(dt,3)/6.0, 
														//pow(dt,4)/8.0, 0, pow(dt,3)/3.0, 0, pow(dt,2)/2.0, 0,
														//0, pow(dt,4)/8.0, 0, pow(dt,3)/3.0, 0, pow(dt,2)/2.0,
														//pow(dt,3)/6.0, 0, pow(dt,2)/2.0, 0, dt*10000, 0,
														//0, pow(dt,3)/6.0, 0, pow(dt,2)/2.0, 0, dt*10000);
	rtracker.processNoiseCov = *(Mat_<float> (6, 6) << 0, 0, 0, 0, 0, 0, 
														0, 0, 0, 0, 0, 0, 
														0, 0, 0, 0, 0, 0,
														0, 0, 0, 0, 0, 0,
														0, 0, 0, 0, dt*100000, 0,
														0, 0, 0, 0, 0, dt*100000);
	rtracker.processNoiseCov = rtracker.processNoiseCov;
	setIdentity(rtracker.measurementNoiseCov, Scalar::all(5));
	setIdentity(rtracker.errorCovPost, Scalar::all(500000));
	setIdentity(rtracker.errorCovPre, Scalar::all(500000));
	
	e1d = 1; e2d = 1; e3d = 1; e4d = 1;
}

HandTracker::~HandTracker()
{
	delete sync;
}

void HandTracker::poseCallback(const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	if (!msg->id.compare(face_found.id))
	{
		pfPose = *msg;
		//lbox.center.x = msg->measurements[0].x;
		//lbox.center.y = msg->measurements[0].y;
		//rbox.center.x = msg->measurements[1].x;
		//rbox.center.y = msg->measurements[1].y;
		//ltracked = true;
		//rtracked = true;
	}
}

void HandTracker::checkHandsInitialisation(cv::Mat likelihood, cv::Mat image3, double xShift,cv::RotatedRect &box, bool &tracked)
{
	/********Left right hand initialisation ************/
	if (!tracked)
	{
		cv::Rect temp;
		temp.x = xShift + image3.cols/16;
		temp.y = image3.rows/8;
		temp.width = image3.cols/2-2*image3.cols/16;
		temp.height = image3.rows - 2*image3.rows/8;
		rectangle(image3, temp, Scalar(255,0,0), 2, 8, 0);
		box = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		box.size.height = min((int)box.size.height,20);
		box.size.width = min((int)box.size.width,20);
		temp = box.boundingRect();
		
		temp = adjustRect(temp,image3.size());
		
		try
		{
			ellipse(image3, box, Scalar(255,0,0), 2, 8);
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
			tracked = false;
		}
		else
		{
			if (xShift > 0)
			{
				rtracker.statePre.at<float>(0) = box.center.x;
				rtracker.statePre.at<float>(1) = box.center.y;
				rtracker.statePre.at<float>(2) = 0;
				rtracker.statePre.at<float>(3) = 0;
				rtracker.statePre.at<float>(4) = 0;
				rtracker.statePre.at<float>(5) = 0;
				setIdentity(rtracker.errorCovPost, Scalar::all(500000));
				setIdentity(rtracker.errorCovPre, Scalar::all(500000));
			}
			else
			{
				ltracker.statePre.at<float>(0) = box.center.x;
				ltracker.statePre.at<float>(1) = box.center.y;
				ltracker.statePre.at<float>(2) = 0;
				ltracker.statePre.at<float>(3) = 0;
				ltracker.statePre.at<float>(4) = 0;
				ltracker.statePre.at<float>(5) = 0;
				setIdentity(ltracker.errorCovPost, Scalar::all(500000));
				setIdentity(ltracker.errorCovPre, Scalar::all(500000));
			}
			tracked = true;
			ellipse(image3, box, Scalar(0,255,0), 2, 8);
		}
	}
	/**********Initialisation using image area **************/
	//~ geometry_msgs::Point ptl;
	//~ ptl.x = likelihood.cols/4.0 + xShift;
	//~ ptl.y = likelihood.rows/2.0;
	//~ ptl.z = 20;
		//~ 
	//~ double lhand_score = cv::sum(likelihood(cv::Rect(ptl.x-25,ptl.y-25,50,50)))[0]/(255.0*50*50);
	//~ if ((lhand_score > lScoreInit)&&(!tracked))
	//~ {
		//~ circle(image3, Point(ptl.x,ptl.y), 50, Scalar(0,255,0), 1, 8, 0);
//~ 
		//~ box = (RotatedRect(Point(ptl.x,ptl.y), Size(50,50),0));
		//~ tracked = true;
	//~ }
	//~ 
	//~ if (!tracked)
	//~ {
		//~ circle(image3, Point(ptl.x,ptl.y), 5, Scalar(255,0,0), 1, 8, 0);
		//~ circle(image3, Point(ptl.x,ptl.y), 10, Scalar(255,0,0), 1, 8, 0);
		//~ circle(image3, Point(ptl.x,ptl.y), 15, Scalar(255,0,0), 1, 8, 0);
		//~ circle(image3, Point(ptl.x,ptl.y), 20, Scalar(255,0,0), 1, 8, 0);
		//~ circle(image3, Point(ptl.x,ptl.y), 25, Scalar(255,0,0), 1, 8, 0);
	//~ }
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

void HandTracker::updateHandPos(cv::Mat likelihood, cv::Mat image3, cv::RotatedRect &box, bool &tracked, face &face_in)
{
	if (tracked)
	{
		box.size.width = box.size.width*2.5;
		box.size.height = box.size.height*2.5;
		cv::Rect temp = box.boundingRect();
		
		try
		{
			ellipse(image3, box, Scalar(0,255,255), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
		
		temp = adjustRect(temp,image3.size());
		box = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		box.size.height = min((int)box.size.height,face_in.roi.height);
		box.size.width = min((int)box.size.width,face_in.roi.width);
		temp = box.boundingRect();
		temp = adjustRect(temp,image3.size());
		tempSR = tempSR*0.6 + 0.4*cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
		ROS_DEBUG("Track: %f",tempSR);
		if ((tempSR < lScoreThresh)||(temp.width <= 1)||(temp.height <= 1))
		{
			ROS_WARN ("Lost Hand! %f %d %d",tempSR,temp.width,temp.height);
			tracked = false;
		}
					
		try
		{
			ellipse(image3, box, Scalar(255,255,0), 2, 8);
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
		
	//cv::Mat roi(likelihood, roi_enlarged);
	//roi = Scalar(0,0,0);
	ellipse(likelihood, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);
	
	cv::Mat tempLikelihoodL,tempLikelihoodR;
	likelihood.copyTo(tempLikelihoodL); 
	if (rtracked)
	{
		// KF prediction
		cv::Mat prediction = rtracker.predict();
		rbox.center.x = prediction.at<float>(0);
		rbox.center.y = prediction.at<float>(1);
		cv::Mat eigenvals, eigenvecs;
		eigen(rtracker.errorCovPre(Range(0,2),Range(0,2)),eigenvals,eigenvecs);
		try
		{
			ellipse(image3,RotatedRect(Point2f(rbox.center.x,rbox.center.y),Size2f(3*sqrt(eigenvals.at<float>(0,0)),3*sqrt(eigenvals.at<float>(1,0))),atan2(eigenvecs.at<float>(0,0),eigenvecs.at<float>(1,0))), Scalar(255,125,125), 1, 8);
		}
		catch (cv::Exception& e )
		{
		}
		// Mask right hand ellipse in left likelihood image
		cv::Mat roiMask = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
		cv::Mat roiMask1 = cv::Mat::ones(image3.rows,image3.cols,CV_8UC1)*255;
		cv::Mat roiMask2 = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
		cv::RotatedRect rbox_t = rbox;
		rbox_t.size.width = rbox_t.size.width*2;
		rbox_t.size.height = rbox_t.size.height*2;
		cv::RotatedRect lbox_t = lbox;
		lbox_t.size.width = lbox_t.size.width*1.0;
		lbox_t.size.height = lbox_t.size.height*1.0;
		try
		{
			ellipse(roiMask1, rbox_t, Scalar(0,0,0), -1, 8);
			ellipse(roiMask2, lbox_t, Scalar(255,255,255), -1, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
		
		bitwise_or(roiMask1,roiMask2,roiMask); // R' + L => Blocks R
		bitwise_and(roiMask,likelihood,tempLikelihoodL);
	}
	
	likelihood.copyTo(tempLikelihoodR); 
	if (ltracked)
	{
		// left hand KF prediction
		cv::Mat prediction = ltracker.predict();
		lbox.center.x = prediction.at<float>(0);
		lbox.center.y = prediction.at<float>(1);
		cv::Mat eigenvals, eigenvecs;
		eigen(ltracker.errorCovPre(Range(0,2),Range(0,2)),eigenvals,eigenvecs);
		try
		{
			ellipse(image3,RotatedRect(Point2f(lbox.center.x,lbox.center.y),Size2f(3*sqrt(eigenvals.at<float>(0,0)),3*sqrt(eigenvals.at<float>(1,0))),atan2(eigenvecs.at<float>(0,0),eigenvecs.at<float>(1,0))), Scalar(255,125,125), 1, 8);
		}
		catch (cv::Exception& e )
		{
		}
		// Mask left hand ellipse in right likelihood image
		cv::Mat roiMask = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
		cv::Mat roiMask1 = cv::Mat::zeros(image3.rows,image3.cols,CV_8UC1);
		cv::Mat roiMask2 = cv::Mat::ones(image3.rows,image3.cols,CV_8UC1)*255;
		cv::RotatedRect rbox_t = rbox;
		rbox_t.size.width = rbox_t.size.width*1.0;
		rbox_t.size.height = rbox_t.size.height*1.0;
		cv::RotatedRect lbox_t = lbox;
		lbox_t.size.width = lbox_t.size.width*2.0;
		lbox_t.size.height = lbox_t.size.height*2.0;
		try
		{
			ellipse(roiMask1, rbox_t, Scalar(255,255,255), -1, 8);
			ellipse(roiMask2, lbox_t, Scalar(0,0,0), -1, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_ERROR("%s",err_msg);
		}	
		bitwise_or(roiMask1,roiMask2,roiMask); // L' + R => Blocks L
		bitwise_and(roiMask,likelihood,tempLikelihoodR);
	}
	
	checkHandsInitialisation(tempLikelihoodL,image3,0,lbox,ltracked);
	updateHandPos(tempLikelihoodL, image3, lbox, ltracked, face_in);
	Mat_<float> measurement(2,1);
	if (ltracked)
	{
		measurement.at<float>(0) = lbox.center.x;
		measurement.at<float>(1) = lbox.center.y;
		cv::Mat estimated = ltracker.correct(measurement);
		lbox.center.x = estimated.at<float>(0);
		lbox.center.y = estimated.at<float>(1);
	}
	
	checkHandsInitialisation(tempLikelihoodR,image3,likelihood.cols/2.0,rbox,rtracked);
	updateHandPos(tempLikelihoodR, image3, rbox, rtracked, face_in);
	if (rtracked)
	{
		measurement.at<float>(0) = rbox.center.x;
		measurement.at<float>(1) = rbox.center.y;
		cv::Mat estimated = rtracker.correct(measurement);
		rbox.center.x = estimated.at<float>(0);
		rbox.center.y = estimated.at<float>(1);
	}
	ROS_DEBUG("Exit: %d %d",ltracked,rtracked);
}

// Gets skin colour likelihood map from face using back projection in Lab
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image2(input);
	cv::Mat image4;
	cvtColor(image2,image4,CV_BGR2Lab);
				
	MatND hist;
	int histSize[] = {60,60};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	
	cv::Rect rec_enlarged;
	rec_enlarged.x = face_in.roi.x+ face_in.roi.width/4;
	rec_enlarged.y = face_in.roi.y+ face_in.roi.height/4;
	rec_enlarged.width = face_in.roi.width - 2*face_in.roi.width/4;
	rec_enlarged.height = face_in.roi.height- 2*face_in.roi.height/4;
	
	pMOG2->operator()(input,fgMaskMOG2,-1);
	
	// Generate output image
	cv::Mat foreground(image4.size(),CV_8UC3,cv::Scalar(255,255,255)); // all white image
	image4.copyTo(foreground,fgMaskMOG2); // bg pixels not copied
	
	cv::Mat subImg1 = image4(rec_enlarged);
	
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist,2,histSize, rangesh, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	hist1 = 0.95*hist1 + 0.05*hist;
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	calcBackProject(&foreground,1,channels,hist1,temp1,rangesh, 1, true);
	
	//GaussianBlur(temp1,temp1,cv::Size(5,5),1,1,BORDER_DEFAULT);
	
	Mat element0 = getStructuringElement( MORPH_ELLIPSE, Size( 2*3 + 1, 2*3+1 ), Point( 3, 3 ) );
	Mat element1 = getStructuringElement( MORPH_ELLIPSE, Size( 2*5 + 1, 2*5+1 ), Point( 5, 5 ) );
	Mat element2 = getStructuringElement( MORPH_ELLIPSE, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
	
	dilate(temp1,temp1,element0);
	erode(temp1,temp1,element1);
	dilate(temp1,temp1,element2);


	cv::Mat image3 = cv::Mat::zeros(image4.rows,image4.cols,CV_8UC3);;//foreground.clone(); // uncomment to view likelihood
	
	//~ 
	Mat dst;
	Canny(image4, dst, 90, 200, 3);
	//~ cvtColor(image3,image3,CV_GRAY2RGB);
		

	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI/180, 10, 6, 10 );
	int e1=0,e2=0,e3=0,e4=0;
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line(image3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
		
		if (pfPose.measurements.size() > 0)
		{
			line(image3, Point(pfPose.measurements[1].x, pfPose.measurements[1].y), Point(pfPose.measurements[5].x, pfPose.measurements[5].y), Scalar(0,255,255), 3, CV_AA);
			line(image3, Point(pfPose.measurements[0].x, pfPose.measurements[0].y), Point(pfPose.measurements[4].x, pfPose.measurements[4].y), Scalar(255,0,255), 3, CV_AA);
			line(image3, Point(pfPose.measurements[7].x, pfPose.measurements[7].y), Point(pfPose.measurements[5].x, pfPose.measurements[5].y), Scalar(0,255,0), 3, CV_AA);
			line(image3, Point(pfPose.measurements[6].x, pfPose.measurements[6].y), Point(pfPose.measurements[4].x, pfPose.measurements[4].y), Scalar(255,255,0), 3, CV_AA);
			double a1  = atan2(l[3]-l[1],l[2]-l[0]);
			double m1 = (l[0]+l[2])/2.0;
			double m2 = (l[1]+l[3])/2.0;
			
			double m3 = (pfPose.measurements[1].x + pfPose.measurements[5].x)/2.0;
			double m4 = (pfPose.measurements[1].y + pfPose.measurements[5].y)/2.0;
			double a2  = atan2(pfPose.measurements[1].y-pfPose.measurements[5].y,pfPose.measurements[1].x-pfPose.measurements[5].x);
			
			double m5 = (pfPose.measurements[0].x + pfPose.measurements[4].x)/2.0;
			double m6 = (pfPose.measurements[0].y + pfPose.measurements[4].y)/2.0;
			double a3  = atan2(pfPose.measurements[0].y-pfPose.measurements[4].y,pfPose.measurements[0].x-pfPose.measurements[4].x);
			
			double m7 = (pfPose.measurements[7].x + pfPose.measurements[5].x)/2.0;
			double m8 = (pfPose.measurements[7].y + pfPose.measurements[5].y)/2.0;
			double a4  = atan2(pfPose.measurements[7].y-pfPose.measurements[5].y,pfPose.measurements[7].x-pfPose.measurements[5].x);
			
			double m9 = (pfPose.measurements[6].x + pfPose.measurements[4].x)/2.0;
			double m10 = (pfPose.measurements[6].y + pfPose.measurements[4].y)/2.0;
			double a5  = atan2(pfPose.measurements[6].y-pfPose.measurements[4].y,pfPose.measurements[6].x-pfPose.measurements[4].x);
			
			//~ ROS_INFO("%f",atan2(sin(a2 - a1),cos(a2 - a1)));
			double d1 = pow(m4-m2,2)+pow(m3-m1,2);
			double d2 = pow(m6-m2,2)+pow(m5-m1,2);
			double d3 = pow(m8-m2,2)+pow(m7-m1,2);
			double d4 = pow(m10-m2,2)+pow(m9-m1,2);
			
			double temp = exp(-0.5*(1.0/125.0*d1 + 1.0/0.01*pow(atan(sin(a2 - a1)/cos(a2 - a1)),2)));//
			//~ e1 = e1 + temp; 
			if (temp > 1e-4)
			{
				e1++;
				line(image3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,255), 1, CV_AA);
				//~ ROS_INFO("Green %f %f",atan(sin(a2 - a1)/cos(a2 - a1))*180/M_PI,d1);
			}		
			
			temp = exp(-0.5*(1.0/125.0*d2 + 1.0/0.01*pow(atan(sin(a3 - a1)/cos(a3 - a1)),2)));// 1.0/sqrt(pow(2*M_PI,3)*pow(125,2)*0.6)*
			if (temp > 1e-4)
			{
				e2++;
				line(image3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,255), 1, CV_AA);
				//~ ROS_INFO("Red %f %f",atan(sin(a3 - a1)/cos(a3 - a1))*180/M_PI,d2);
			}
			//~ e2 = e2 + temp;
			
			temp = exp(-0.5*(1.0/125.0*d3 + 1.0/0.01*pow(atan(sin(a4 - a1)/cos(a4 - a1)),2)));// 1.0/sqrt(pow(2*M_PI,3)*pow(125,2)*0.6)*
			if (temp > 1e-4)
			{
				e3++;
				line(image3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 1, CV_AA);
				//~ ROS_INFO("Red %f %f",atan(sin(a3 - a1)/cos(a3 - a1))*180/M_PI,d2);
			}
			//~ e3 = e3 + temp;
			
			temp = exp(-0.5*(1.0/125.0*d4 + 1.0/0.01*pow(atan(sin(a5 - a1)/cos(a5 - a1)),2)));// 1.0/sqrt(pow(2*M_PI,3)*pow(125,2)*0.6)*
			if (temp > 1e-4)
			{
				e4++;
				line(image3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,0), 1, CV_AA);
				//~ ROS_INFO("Red %f %f",atan(sin(a3 - a1)/cos(a3 - a1))*180/M_PI,d2);
			}
			//~ e2 = e2 + temp;
			
			
		}
	}
	if (pfPose.measurements.size() > 0)
	{
		pfPose.measurements.clear();
		pfPose.names.clear();
		e1d = 0.75*e1d+0.25*e1;
		e2d = 0.75*e2d+0.25*e2;
		e3d = 0.75*e3d+0.25*e3;
		e4d = 0.75*e4d+0.25*e4;
		
		if ((e1d < 1)||(e3d < 1))
		{
			rtracked = false;
			ROS_WARN("Reset (line check fail): %f %f",e1d,e3d);
			e1d = 1;
			e3d = 1;
		}
		
		if ((e2d < 1)||(e4d < 1))
		{
			ltracked = false;
			ROS_WARN("Reset (line check fail): %f %f",e2d,e4d);
			e2d = 1;
			e4d = 1;
		}
		ROS_DEBUG("Forearm evidence: %f %f %f %f",e1d,e2d,e3d,e4d);
	}
	
	ROS_DEBUG("Entry: %d %d",ltracked,rtracked);

	// Detect hands and update pose estimate
	HandDetector(temp1,face_in,image3);
				
	// Draw face rectangles on display image
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(rec_enlarged.x,rec_enlarged.y), Point(rec_enlarged.x+rec_enlarged.width,rec_enlarged.y+rec_enlarged.height), Scalar(0,255,0), 4, 8, 0);
	
	cv_bridge::CvImage img2;
	img2.encoding = "mono8";
	img2.image = temp1;			
	likelihood_pub.publish(img2.toImageMsg()); // publish result image
	
	return image3;
	//~ return foreground;
	//return foreground;
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
			ltracked = false;
			rtracked = false;
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
	//~ ROS_INFO ("Message received");
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
			
		updateFaceInfo(msg); // update face list
		
		cv::Mat outputImage = cv::Mat::zeros(image.rows,image.cols,image.type()); // display image
		if (face_found.views > 0) // get hands
		{
			outputImage = getHandLikelihood(image,face_found);
		}
		cv_bridge::CvImage img2;
		img2.encoding = "rgb8";
		img2.header = immsg->header;
		img2.image = outputImage;			
		pub.publish(img2.toImageMsg()); // publish result image
		
		handBlobTracker::HFPose2D rosHands;
		handBlobTracker::HFPose2DArray rosHandsArr;
		rosHands.x = lbox.center.x;
		rosHands.y = lbox.center.y;
		rosHandsArr.measurements.push_back(rosHands);
		rosHandsArr.names.push_back("Left Hand");
		rosHands.x = rbox.center.x;
		rosHands.y = rbox.center.y;
		rosHandsArr.measurements.push_back(rosHands);
		rosHandsArr.names.push_back("Right Hand");
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + int(face_found.roi.height/2.0);
		rosHandsArr.measurements.push_back(rosHands);
		rosHandsArr.names.push_back("Head");
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + 3.25/2.0*face_found.roi.height;
		rosHandsArr.measurements.push_back(rosHands); //Neck
		rosHandsArr.names.push_back("Neck");
		rosHandsArr.valid.push_back(ltracked);
		rosHandsArr.valid.push_back(rtracked);
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
