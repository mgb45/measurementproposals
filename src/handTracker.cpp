#include "handTracker.h"

using namespace cv;
using namespace std;

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/handImage",1); //ROS
	hand_face_pub = nh.advertise<handBlobTracker::HFPose2DArray>("/faceHandPose", 1000);
	
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, roi_sub);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
	rtracked = false;
	ltracked = false;
	double dt = 0.1;
	
	ltracker.init(4,2,0,CV_32F);
	ltracker.transitionMatrix = *(Mat_<float> (4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
	setIdentity(ltracker.measurementMatrix);
	setIdentity(ltracker.processNoiseCov, Scalar::all(100));
	ltracker.processNoiseCov.at<float>(0,0) = 0;
	ltracker.processNoiseCov.at<float>(1,1) = 0;
	setIdentity(ltracker.measurementNoiseCov, Scalar::all(15));
	setIdentity(ltracker.errorCovPost, Scalar::all(2));
	
	rtracker.init(4,2,0,CV_32F);
	rtracker.transitionMatrix = *(Mat_<float> (4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
	setIdentity(rtracker.measurementMatrix);
	setIdentity(rtracker.processNoiseCov, Scalar::all(100));
	rtracker.processNoiseCov.at<float>(0,0) = 0;
	rtracker.processNoiseCov.at<float>(1,1) = 0;
	setIdentity(rtracker.measurementNoiseCov, Scalar::all(15));
	setIdentity(rtracker.errorCovPost, Scalar::all(2));
	
	timePre = ros::Time::now().toSec();
}

cv::Mat HandTracker::predictPos(cv::KalmanFilter &tracker)
{
	tracker.transitionMatrix.at<float>(0,2) = dt;
	return tracker.predict();
	//~ roi.x = std::max((int)prediction.at<float>(0),0);
	//~ roi.x = std::min(roi.x,640-roi.width);
	//~ roi.y = std::max((int)prediction.at<float>(1),0);
	//~ roi.y = std::min(roi.y,480-roi.height);
}

cv::Mat HandTracker::updatePos(float x, float y, cv::KalmanFilter &tracker)
{
	Mat_<float> measurement(2,1);
	measurement.at<float>(0) = x;
	measurement.at<float>(1) = y;
	return tracker.correct(measurement);
}

HandTracker::~HandTracker()
{
	delete sync;
}

// Hand detector: given likelihood of skin, detects hands and intialising area and uses camshift to track them
void HandTracker::HandDetector(cv::Mat likelihood, face &face_in, cv::Mat image3)
{
	//Set face probability to zero, not a hand
	int pre_height = face_in.roi.height;
	face_in.roi.height = min(face_in.roi.height*2,image3.rows-face_in.roi.y);
	
	cv::Mat roi(likelihood, face_in.roi);
	roi = Scalar(0,0,0);
	face_in.roi.height = pre_height;
	
	cv::Mat image_or = image3.clone();
	
	cv::Mat hsv;
	cvtColor(image_or, hsv, CV_BGR2HSV);
	vector<Mat> bgr_planes;
	split(hsv, bgr_planes);
	
	// check if hand near intialisation area left
	geometry_msgs::Point ptl;
	ptl.x = likelihood.cols/4.0;
	ptl.y = likelihood.rows/2.0;
	ptl.z = 20;
	
	
	double lhand_score = cv::sum(likelihood(cv::Rect(ptl.x-25,ptl.y-25,50,50)))[0]/(255.0*50*50);
	if ((lhand_score > 0.2)&&(!ltracked))
	{
		circle(image3, Point(ptl.x,ptl.y), 50, Scalar(0,255,0), 1, 8, 0);
		ltracked = true;

		lbox = RotatedRect(Point(ptl.x,ptl.y), Size(50,50),0);
		ltracker.statePost.at<float>(0) = ptl.x;
		ltracker.statePost.at<float>(1) = ptl.y;
		ltracker.statePost.at<float>(2) = 0;
		ltracker.statePost.at<float>(3) = 0;
		setIdentity(ltracker.errorCovPost, Scalar::all(5));
	}
	
	
	// check if hand near intialisation area right
	geometry_msgs::Point ptr;
	ptr.x = likelihood.cols/2.0 + likelihood.cols/4.0;
	ptr.y = likelihood.rows/2.0;
	ptr.z = 20;
	double rhand_score = cv::sum(likelihood(cv::Rect(ptr.x-25,ptr.y-25,50,50)))[0]/(255.0*50*50);
			
	if ((rhand_score > 0.2)&&(!rtracked))
	{
		circle(image3, Point(ptr.x,ptr.y), 50, Scalar(0,255,0), 1, 8, 0);
		rtracked = true;
		
		rbox = RotatedRect(Point(ptr.x,ptr.y), Size(50,50),0);
		rtracker.statePost.at<float>(0) = ptr.x;
		rtracker.statePost.at<float>(1) = ptr.y;
		rtracker.statePost.at<float>(2) = 0;
		rtracker.statePost.at<float>(3) = 0;
		setIdentity(rtracker.errorCovPost, Scalar::all(5));
	}
			
	ROS_INFO("Hand scores: %f %f",lhand_score,rhand_score);
	
	if (rtracked)
	{
		//~ Mat prediction = predictPos(rtracker);
		//~ rbox.center.x = (int)prediction.at<float>(0);
		//~ rbox.center.y = (int)prediction.at<float>(1);
		cv::Rect temp = rbox.boundingRect();
		temp.x = min(temp.x+int(double(temp.width)/2.0)-int(face_in.roi.width/2.0),temp.x);
		temp.y = min(temp.y+int(double(temp.height)/2.0)-int(face_in.roi.height/2.0),temp.y);
		temp.width = min(temp.width,face_in.roi.width);
		temp.height = min(temp.height,face_in.roi.height);
		
		temp.x = min(max(temp.x,1),image3.cols);
		temp.y = min(max(temp.y,1),image3.rows);
		temp.width = min(temp.width,image3.cols-temp.x);
		temp.height = min(temp.height,image3.rows-temp.y);
		rbox = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		
		temp = rbox.boundingRect();
		temp.x = min(max(temp.x,1),image3.cols);
		temp.y = min(max(temp.y,1),image3.rows);
		temp.width = min(temp.width,image3.cols-temp.x);
		temp.height = min(temp.height,image3.rows-temp.y);			
		double tempS = cv::sum(likelihood(temp))[0]/(255.0*temp.width*temp.height);
		ROS_INFO("%f",tempS);
		if ((tempS < 0.01)||(temp.width <= 5)||(temp.height <= 5))
		{
			rtracked = false;
		}
		//~ Mat estimate = updatePos(rbox.center.x,rbox.center.y,rtracker);
		//~ rbox.center.x = (int)estimate.at<float>(0);
		//~ rbox.center.y = (int)estimate.at<float>(1);
					
		try
		{
			ellipse(image3, rbox, Scalar(255,255,0), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_INFO("%s",err_msg);
		}	
	}
	else
	{
		circle(image3, Point(ptr.x,ptr.y), 5, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptr.x,ptr.y), 10, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptr.x,ptr.y), 15, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptr.x,ptr.y), 20, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptr.x,ptr.y), 25, Scalar(255,0,0), 1, 8, 0);
		//~ rtracked = false;
	}
	
	if (ltracked)
	{
		//~ Mat prediction = predictPos(ltracker);
		//~ lbox.center.x = (int)prediction.at<float>(0);
		//~ lbox.center.y = (int)prediction.at<float>(1);
		cv::Rect temp = lbox.boundingRect();
		
		temp.x = min(temp.x+int(double(temp.width)/2.0)-int(face_in.roi.width/2.0),temp.x);
		temp.y = min(temp.y+int(double(temp.height)/2.0)-int(face_in.roi.height/2.0),temp.y);
		temp.width = min(temp.width,face_in.roi.width);
		temp.height = min(temp.height,face_in.roi.height);
		
		temp.x = min(max(temp.x,1),image3.cols);
		temp.y = min(max(temp.y,1),image3.rows);
		temp.width = min(temp.width,image3.cols-temp.x);
		temp.height = min(temp.height,image3.rows-temp.y);
		lbox = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		
		temp = lbox.boundingRect();
		temp.x = min(max(temp.x,1),image3.cols);
		temp.y = min(max(temp.y,1),image3.rows);
		temp.width = min(temp.width,image3.cols-temp.x);
		temp.height = min(temp.height,image3.rows-temp.y);			
		double tempS = cv::sum(likelihood(temp))[0]/(255.0*temp.width*temp.height);
		ROS_INFO("%f",tempS);
		if ((tempS < 0.01)||(temp.width <= 5)||(temp.height <= 5))
		{
			ltracked = false;
		}
		
		//~ Mat estimate = updatePos(lbox.center.x,lbox.center.y,ltracker);
		//~ lbox.center.x = (int)estimate.at<float>(0);
		//~ lbox.center.y = (int)estimate.at<float>(1);
		try
		{
			ellipse(image3, lbox, Scalar(255,255,0), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
		}	
	}
	else
	{
		circle(image3, Point(ptl.x,ptl.y), 5, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptl.x,ptl.y), 10, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptl.x,ptl.y), 15, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptl.x,ptl.y), 20, Scalar(255,0,0), 1, 8, 0);
		circle(image3, Point(ptl.x,ptl.y), 25, Scalar(255,0,0), 1, 8, 0);
	}
}

// Gets skin colour likelihood map from face using back projection in HSV
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image2(input);
	cv::Mat image4;
	cvtColor(image2,image4,CV_BGR2Lab);
							
	vector<Mat> bgr_planes;
	split(image4, bgr_planes);
				
	MatND hist1, hist2;
	int histSize = 50;
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh = {h_range};
	const float* rangess = {s_range};
	
	cv::Rect rec_enlarged;
	rec_enlarged.x = face_in.roi.x+ face_in.roi.width/4;
	rec_enlarged.y = face_in.roi.y+ face_in.roi.height/4;
	rec_enlarged.width = face_in.roi.width - 2*face_in.roi.width/4;
	rec_enlarged.height = face_in.roi.height- 2*face_in.roi.height/4;
	cv::Mat subImg1 = bgr_planes[1](rec_enlarged);
	//~ medianBlur(subImg1,subImg1,7);
	cv::Mat subImg2 = bgr_planes[2](rec_enlarged);
	//~ medianBlur(subImg2,subImg2,7);
	calcHist(&subImg1, 1, 0, Mat(), hist1, 1, &histSize, &rangesh, true, false);
	calcHist(&subImg2, 1, 0, Mat(), hist2, 1, &histSize, &rangess, true, false);
	normalize(hist1, hist1, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(hist2, hist2, 0, 255, NORM_MINMAX, -1, Mat());
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	cv::Mat temp2(input.rows,input.cols,CV_64F);
	calcBackProject(&bgr_planes[1], 1, 0, hist1, temp1, &rangesh, 1, true);
	calcBackProject(&bgr_planes[2], 1, 0, hist2, temp2, &rangess, 1, true);
	
	bitwise_and(temp1,temp2,temp1,Mat());
	
	//~ medianBlur(temp1,temp1,7);
	GaussianBlur(temp1,temp1,cv::Size(5,5),1,1,BORDER_DEFAULT);
//~ RNG rng(12345);
   //~ vector<vector<Point> > contours;
  //~ vector<Vec4i> hierarchy;	
	//~ /// Detect edges using canny
	//~ Canny(temp1, temp1, 100, 200, 3 );
	//~ /// Find contours
	//~ findContours(temp1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//~ 
  //~ /// Draw contours
  //~ for( int i = 0; i< contours.size(); i++ )
     //~ {
       //~ Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       //~ drawContours( temp1, contours, i, color, 2, 0, hierarchy, 0, Point() );
     //~ }
	
	//~ cvtColor(temp1,temp1,CV_GRAY2RGB);
		
	//~ cv::Mat image3 = image2.clone();
	cv::Mat image3 = temp1.clone(); // uncomment to view likelihood
	cvtColor(image3,image3,CV_GRAY2RGB);
	// Detect hands and update pose estimate
	HandDetector(temp1,face_in,image3);
				
	// Draw face rectangles on display image
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(rec_enlarged.x,rec_enlarged.y), Point(rec_enlarged.x+rec_enlarged.width,rec_enlarged.y+rec_enlarged.height), Scalar(0,255,0), 4, 8, 0);
	//~ rectangle(image3, Point(face_in.roi.x+ face_in.roi.width/5.0,face_in.roi.y+ face_in.roi.height/5.0), Point(face_in.roi.x+4*face_in.roi.width/5.0,face_in.roi.y+4*face_in.roi.height/5.0), Scalar(0,255,0), 4, 8, 0);
	
	return image3;
}

// Update tracked face with latest info
void HandTracker::updateFaceInfo(const faceTracking::ROIArrayConstPtr& msg)
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

// image and face roi callback
void HandTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg)
{
	ROS_INFO ("Message received");
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
		
		dt  = msg->header.stamp.toSec() - timePre;
		timePre = msg->header.stamp.toSec();
			
		updateFaceInfo(msg); // update face list
		
		cv::Mat outputImage = cv::Mat::zeros(image.rows,image.cols,image.type()); // display image
		if (face_found.views > 0) // get hands
		{
			outputImage = getHandLikelihood(image,face_found);
		}
		cv_bridge::CvImage img2;
		img2.encoding = "rgb8";
		img2.image = outputImage;			
		pub.publish(img2.toImageMsg()); // publish result image
		
		if (ltracked||rtracked)
		{
			handBlobTracker::HFPose2D rosHands;
			handBlobTracker::HFPose2DArray rosHandsArr;
			rosHands.x = lbox.center.x;
			rosHands.y = lbox.center.y;
			rosHandsArr.measurements.push_back(rosHands);
			rosHands.x = rbox.center.x;
			rosHands.y = rbox.center.y;
			rosHandsArr.measurements.push_back(rosHands);
			rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
			rosHands.y = face_found.roi.y + int(face_found.roi.width/2.0);
			rosHandsArr.measurements.push_back(rosHands);
			rosHandsArr.valid.push_back(ltracked);
			rosHandsArr.valid.push_back(rtracked);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.header = msg->header;
			rosHandsArr.id = face_found.id;
			hand_face_pub.publish(rosHandsArr);
		}
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}
