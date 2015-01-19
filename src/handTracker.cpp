#include "handTracker.h"

using namespace cv;
using namespace std;

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/likelihood",1); //ROS
	hand_face_pub = nh.advertise<measurementproposals::HFPose2DArray>("/faceHandPose", 10);
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;

	cv::Mat subImg1 = cv::Mat::zeros(50,50,CV_8UC3);
	
	int histSize[] = {35,35};
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

// Gets skin colour likelihood map from face using back projection in a-b channels of Lab colour space
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image4;
	cvtColor(input,image4,CV_BGR2Lab);
				
	MatND hist;
	int histSize[] = {35,35};
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
	
	// Mask face area
	cv::Rect roi_enlarged; // enlarge face to cover neck and ear blobs
	roi_enlarged.height = face_in.roi.height*1.8;
	roi_enlarged.width = face_in.roi.width*1.2;
	roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	roi_enlarged.y = face_in.roi.height/2 + face_in.roi.y - roi_enlarged.height/2;
			
	ellipse(temp1, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);

	return temp1;
}

// Propose measurements from liklihood
std::vector<int> HandTracker::proposeMeasurements(cv::Mat L_image, int N)
{
	std::vector<int> bins;
	// Convert likelihood to weights
	Mat likelihood;
	threshold(L_image, L_image, 255.0/2.0, 255.0, cv::THRESH_BINARY);
	if (cv::sum(L_image)[0]/(255.0*L_image.rows*L_image.cols)<1e-4)
	{
		return bins;
	}
	
	L_image.convertTo(likelihood, CV_64F, 1.0/255.0, 0);
	Mat vec = likelihood.reshape(0,1);
	vec = vec/sum(vec)[0];
	
	std::vector<double> weights(vec);
	bins = resample(weights, N);
	
	return bins;//measurements;
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

// Return best weight
double HandTracker::maxWeight(std::vector<double> weights)
{
	double mw = 0;
	for (int i = 0; i < (int)weights.size(); i++)
	{
		if (weights[i] > mw)
		{
			mw = weights[i];
		}
	}
	return mw;
}

// Resample according to weights
std::vector<int> HandTracker::resample(std::vector<double> weights, int N)
{
	int L = weights.size();
	std::vector<int> indicators;
	int idx = rand() % L;
	double beta = 0.0;
	double mw = maxWeight(weights);
	cv::RNG rng(cv::getTickCount());
	if (mw == 0)
	{
		weights.clear();
		for (int i = 0; i < N; i++)
		{
			indicators.push_back(rng.uniform(0, L));
			weights.push_back(1.0/(double)N);
		}
	}
	else
	{
		idx = 0;
		double step = 1.0 / (double)N;
		beta = rng.uniform(0.0, 1.0)*step;
		for (int i = 0; i < N; i++)
		{
			while (beta > weights[idx])
			{
				beta -= weights[idx];
				idx = (idx + 1) % L;
			}
			beta += step;
			indicators.push_back(idx);
		}
	}
	return indicators;
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
			int N = 1500;
			outputImage = getHandLikelihood(image,face_found);
			
			// Only get measurements if entropy of likelihood image is low enough.
			//cv::Mat ent, entl;
			//int histSize = 255;
			//float h_range[] = {0, 255};
			//const float *rangesh = {h_range};
			////calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, rangesh, true, false);
			//calcHist(&outputImage,1,0,Mat(),ent,1,&histSize, &rangesh, true, false);
			//normalize(ent, ent, 0, 1, NORM_MINMAX, -1, Mat());
			//cv::log(ent,entl);
			//ent = entl.mul(ent);
			
			std::vector<int> bins = proposeMeasurements(outputImage,N);
			if ((int)bins.size() > 0)
			{
				measurementproposals::HFPose2D rosMeas;
				measurementproposals::HFPose2DArray rosMeasArr;
				// Add head measurement to array
				rosMeas.x = face_found.roi.x + int(face_found.roi.width/2.0);
				rosMeas.y = face_found.roi.y + int(face_found.roi.height/2.0);
				rosMeasArr.measurements.push_back(rosMeas);
				// Add neck measurement to array
				rosMeas.x = face_found.roi.x + int(face_found.roi.width/2.0);
				rosMeas.y = face_found.roi.y + 3.25/2.0*face_found.roi.height;
				rosMeasArr.measurements.push_back(rosMeas);
				// Add potential hand measurements to array
				Point pt;
				for (int i = 0; i < N; i ++)
				{
					pt.y = bins[i]/image.cols;
					pt.x = bins[i]%image.cols;
					circle(image, pt, 2, Scalar(0,0,255), -1, 8);
					rosMeas.x = pt.x;
					rosMeas.y = pt.y;
					rosMeasArr.measurements.push_back(rosMeas);
				}
				rosMeasArr.header = msg->header;
				rosMeasArr.id = face_found.id;
				hand_face_pub.publish(rosMeasArr);
			}
		}
		else
		{
			measurementproposals::HFPose2DArray rosMeasArr;
			rosMeasArr.header = msg->header;
			rosMeasArr.id = '0';
			hand_face_pub.publish(rosMeasArr);
		}
	
		cv_bridge::CvImage img2;
		img2.encoding = "rgb8";
		img2.header = immsg->header;
		img2.image = image;			
		pub.publish(img2.toImageMsg()); // publish result image
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}
