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
	pose_sub = nh.subscribe("/correctedFaceHandPose", 1,&HandTracker::poseCallback,this); // requires face array input
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, roi_sub);
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
}

HandTracker::~HandTracker()
{
	delete sync;
}

void HandTracker::poseCallback(const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	if (!msg->id.compare(face_found.id))
	{
		ROS_INFO("Hand_feedback");
		lbox.center.x = msg->measurements[0].x;
		lbox.center.y = msg->measurements[0].y;
		rbox.center.x = msg->measurements[1].x;
		rbox.center.y = msg->measurements[1].y;
		ltracked = true;
		rtracked = true;
	}
}

void HandTracker::checkHandsInitialisation(cv::Mat likelihood, cv::Mat image3, double xShift,cv::RotatedRect &box, bool &tracked)
{
	if (!tracked)
	{
		cv::Rect temp;
		temp.x = xShift;
		temp.y = 0;
		temp.width = image3.cols/2;
		temp.height = image3.rows/2;
		box = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		box.size.height = min((int)box.size.height,50);
		box.size.width = min((int)box.size.width,50);
		temp = box.boundingRect();
		
		//~ temp.x = temp.x+int(double(temp.width)/2.0)-int(face_in.roi.width/2.0);//,temp.x);
		//~ temp.y = temp.y+int(double(temp.height)/2.0)-int(face_in.roi.height/2.0);//),temp.y);
		//~ temp.width = min(temp.width,face_in.roi.width);
		//~ temp.height = min(temp.height,face_in.roi.height);
		//~ 
		temp = adjustRect(temp,image3.size());
		tempSR = tempSR*0.8 + 0.2*cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
		ROS_INFO("%f",tempSR);
		if ((tempSR < lScoreInit)||(temp.width <= 5)||(temp.height <= 5))
		{
			tracked = false;
		}
		else
		{
			box = (RotatedRect(Point(box.center.x,box.center.y), Size(50,50),0));
			tracked = true;
		}
	}
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
	newRect.x = min(max(temp.x,1),size.width);
	newRect.y = min(max(temp.y,1),size.height);
	newRect.width = min(temp.width,size.width-temp.x);
	newRect.height = min(temp.height,size.height-temp.y);
	return newRect;
}


void HandTracker::updateHandPos(cv::Mat likelihood, cv::Mat image3, cv::RotatedRect &box, bool &tracked, face &face_in)
{
	if (tracked)
	{
		box.size.width = box.size.width*1.5;
		box.size.height = box.size.height*1.5;
		cv::Rect temp = box.boundingRect();
		
		try
		{
			ellipse(image3, box, Scalar(0,255,255), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_INFO("%s",err_msg);
		}	
		
		//~ temp.x = temp.x+int(double(temp.width)/2.0)-int(face_in.roi.width/1.0);//,temp.x);
		//~ temp.y = temp.y+int(double(temp.height)/2.0)-int(face_in.roi.height/1.0);//,temp.y);
		//~ temp.width = min(temp.width,2*face_in.roi.width);
		//~ temp.height = min(temp.height,2*face_in.roi.height);
		
		temp = adjustRect(temp,image3.size());
		box = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
		
		box.size.height = min((int)box.size.height,face_in.roi.height);
		box.size.width = min((int)box.size.width,face_in.roi.width);
		temp = box.boundingRect();
		
		//~ temp.x = temp.x+int(double(temp.width)/2.0)-int(face_in.roi.width/2.0);//,temp.x);
		//~ temp.y = temp.y+int(double(temp.height)/2.0)-int(face_in.roi.height/2.0);//),temp.y);
		//~ temp.width = min(temp.width,face_in.roi.width);
		//~ temp.height = min(temp.height,face_in.roi.height);
		//~ 
		temp = adjustRect(temp,image3.size());
		tempSR = tempSR*0.8 + 0.2*cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
		ROS_INFO("%f",tempSR);
		if ((tempSR < lScoreThresh)||(temp.width <= 5)||(temp.height <= 5))
		{
			tracked = false;
		}
					
		try
		{
			ellipse(image3, box, Scalar(255,255,0), 2, 8);
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			ROS_INFO("%s",err_msg);
		}	
	}
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
	

	checkHandsInitialisation(likelihood,image3,0,lbox,ltracked);
	checkHandsInitialisation(likelihood,image3,likelihood.cols/2.0,rbox,rtracked);
	
	cv::Mat tempLikelihood;
	likelihood.copyTo(tempLikelihood); // Mask left hand ellipse
	if (ltracked)
	{
		cv::Mat roi1(tempLikelihood, adjustRect(lbox.boundingRect(),image3.size()));
		roi1 = Scalar(0,0,0);
	}
	updateHandPos(tempLikelihood, image3, rbox, rtracked, face_in);
	likelihood.copyTo(tempLikelihood); // Mask right hand ellipse
	if (rtracked)
	{
		cv::Mat roi2(tempLikelihood, adjustRect(rbox.boundingRect(),image3.size()));
		roi2 = Scalar(0,0,0);
	}
	updateHandPos(tempLikelihood, image3, lbox, ltracked, face_in);
	
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
	
	pMOG2->operator()(input,fgMaskMOG2,-10);
	
	//~ cv::Mat result; // segmentation (4 possible values)
   //~ cv::Mat bgModel,fgModel; // the models (internally used)
   //~ // GrabCut segmentation
   //~ cv::grabCut(input,    // input image
            //~ result,      // segmentation result
            //~ face_in.roi,   // rectangle containing foreground 
            //~ bgModel,fgModel, // models
            //~ 5,           // number of iterations
            //~ cv::GC_INIT_WITH_RECT); // use rectangle
            //~ 
            //~ // Get the pixels marked as likely foreground
   //~ cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
                //~ 
    //~ 
      //~ // Generate output image
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

	cv::Mat image3 = temp1.clone(); // uncomment to view likelihood
	cvtColor(image3,image3,CV_GRAY2RGB);

	// Detect hands and update pose estimate
	HandDetector(temp1,face_in,image3);
				
	// Draw face rectangles on display image
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(rec_enlarged.x,rec_enlarged.y), Point(rec_enlarged.x+rec_enlarged.width,rec_enlarged.y+rec_enlarged.height), Scalar(0,255,0), 4, 8, 0);
	
	return image3;
	//~ return foreground;
	//return foreground;
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
			rosHands.y = face_found.roi.y + int(face_found.roi.height/2.0);
			rosHandsArr.measurements.push_back(rosHands);
			rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
			rosHands.y = face_found.roi.y + 3.0/2.0*face_found.roi.height;
			rosHandsArr.measurements.push_back(rosHands); //Neck
			rosHandsArr.valid.push_back(ltracked);
			rosHandsArr.valid.push_back(rtracked);
			rosHandsArr.valid.push_back(true);
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
