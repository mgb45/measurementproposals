#include "blobTracker.h"

using namespace cv;
using namespace std;

// Constructer: Body tracker 
BlobTracker::BlobTracker()
{
	namedWindow("win0", CV_WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow("win1", CV_WINDOW_AUTOSIZE );// Create a window for display.
	
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/blobImage",1); //ROS
	
	//gmm1 = new cv::EM(5, cv::EM::COV_MAT_GENERIC, cv::TermCriteria(1, 0.1, CV_TERMCRIT_ITER|CV_TERMCRIT_EPS));
	//~ gmm_model = new GMM(255); // Skin colour gmm
	im_received = false;
	
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, roi_sub);
	sync->registerCallback(boost::bind(&BlobTracker::callback, this, _1, _2));
	
	params = new SimpleBlobDetector::Params; // Opencv blob detector object
	
    params->minArea = 400; // min area of hand blobs
    params->maxArea = 10000; // max area of hand blobs
    	
    // Locate hand blobs by area only
	params->filterByColor = false;
    params->filterByCircularity = false;
    params->filterByArea = true;
    params->filterByInertia = false;
    params->filterByConvexity = false;
}

BlobTracker::~BlobTracker()
{
	//~ delete gmm_model;
	//delete gmm1;
	delete sync;
}

// blob detector: given input image and gmm skin model, extracts likely hand positions and performs particle filter update to estimate upper body joint positions
void BlobTracker::blobDetector(cv::Mat output, body &body_in, cv::Mat image3)
{
   // Set blob detector parameters based on detected face size.
    double fmin = 100000, fmax = 0;
    for (int i = 0; i < (int)faces.size(); i++)
    {
		fmin = std::min(fmin,(double)std::min(faces[i].roi.height,faces[i].roi.width));
		fmax = std::max(fmax,(double)std::max(faces[i].roi.height,faces[i].roi.width));
	}
    params->minArea = M_PI*fmin*fmin/100.0;
    params->maxArea = M_PI*fmax*fmax;
    params->minDistBetweenBlobs = 5.0*fmin/6.0;

    params->minThreshold = 110;
    params->maxThreshold = 255;
    params->thresholdStep = 15;
    
    SimpleBlobDetector blobDetector(*params);
    blobDetector.create("SimpleBlob");

	// Find blob locations
    vector<KeyPoint> keyPoints;
    blobDetector.detect(output, keyPoints);
    
    if ((int)keyPoints.size() >= 2) // if at least 2 blobs found
    {
		int minIdx1 = -1;
		double minD1 = 200;
		int minIdx2 = -1; 
		double minD2 = 200;
		for (int i = 0; i < (int)keyPoints.size(); i++) // iterate through detected blobs
		{
			geometry_msgs::Point pt1;
			pt1.x = body_in.roi.x + body_in.roi.width/2.0;
			pt1.y = body_in.roi.y + body_in.roi.height/2.0;
			pt1.z = body_in.roi.height;
			if (blobDist(pt1,keyPoints[i]) > body_in.roi.width) // if blob not near head
			{
				if (body_in.leftHand.z != -1) // if hand already tracked
				{
					double temp = blobDist(body_in.leftHand,keyPoints[i]);
					if (minD1 > temp) // and distance between existing hand position and blob small enough
					{
						minD1 = temp; // set blob as left hand
						minIdx1 = i;
					}
				}
				else // check if hand near intialisation area
				{
					
					geometry_msgs::Point pt1;
					pt1.x = output.cols/4.0;
					pt1.y = output.rows/2.0;
					pt1.z = 20;
					circle(image3, Point(pt1.x,pt1.y), 5, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 10, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 15, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 20, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 25, Scalar(255,0,0), 1, 8, 0);
					double temp = blobDist(pt1,keyPoints[i]);
					if (50 > temp)
					{
						minD1 = temp; // set blob as left hand
						minIdx1 = i;
					}
				}
				
				if (body_in.rightHand.z != -1) // Do same for right hand
				{
					double temp = blobDist(body_in.rightHand,keyPoints[i]);
					if (50 > temp)
					{
						minD2 = temp;
						minIdx2 = i;
					}
				}
				else
				{
					geometry_msgs::Point pt1;
					pt1.x = output.cols/2.0 + output.cols/4.0;
					pt1.y = output.rows/2.0;
					pt1.z = 20;
					circle(image3, Point(pt1.x,pt1.y), 5, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 10, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 15, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 20, Scalar(255,0,0), 1, 8, 0);
					circle(image3, Point(pt1.x,pt1.y), 25, Scalar(255,0,0), 1, 8, 0);
					double temp = blobDist(pt1,keyPoints[i]);
					if (minD2 > temp)
					{
						minD2 = temp;
						minIdx2 = i;
					}
				}
				ellipse(image3, keyPoints[i].pt, Size(keyPoints[i].size,keyPoints[i].size), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0); //draw ellipse for blob
			}
		}
		// If suitable hands found, update body pose estimate
		if ((minIdx1 != -1)&&(minIdx2 != -1)&&(minIdx1!=minIdx2))
		{
			cv::Mat measurement1(2,2,CV_64F);
			measurement1.at<double>(0,0) = body_in.roi.x + body_in.roi.width/2.0;
			measurement1.at<double>(0,1) = body_in.roi.y + body_in.roi.height/2.0;
			measurement1.at<double>(1,0) = keyPoints[minIdx1].pt.x;
			measurement1.at<double>(1,1) = keyPoints[minIdx1].pt.y;
			body_in.pf1->update(measurement1); // particle filter measurement left arm
			
			cv::Mat measurement2(2,2,CV_64F);
			measurement2.at<double>(0,0) = body_in.roi.x + body_in.roi.width/2.0;
			measurement2.at<double>(0,1) = body_in.roi.y + body_in.roi.height/2.0;
			measurement2.at<double>(1,0) = keyPoints[minIdx2].pt.x;
			measurement2.at<double>(1,1) = keyPoints[minIdx2].pt.y;
			body_in.pf2->update(measurement2); // particle filter measurement right arm
			
			cv::Mat e1 = body_in.pf1->getEstimator(); // Weighted average pose estimate
			cv::Mat e2 = body_in.pf2->getEstimator();
			
			// Set new hand positions to particle filter pose estimates
			body_in.leftHand.x = e1.at<double>(0,0);
			body_in.leftHand.y = e1.at<double>(0,1);
			body_in.leftHand.z = keyPoints[minIdx1].size; 
			
			body_in.rightHand.x = e2.at<double>(0,0);
			body_in.rightHand.y = e2.at<double>(0,1);
			body_in.rightHand.z = keyPoints[minIdx2].size;
			
			// Draw body lines
			//h-e
			line(image3, Point(e1.at<double>(0,0),e1.at<double>(0,1)), Point(e1.at<double>(0,2),e1.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
			line(image3, Point(e2.at<double>(0,0),e2.at<double>(0,1)), Point(e2.at<double>(0,2),e2.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
			//E -S
			line(image3, Point(e1.at<double>(0,2),e1.at<double>(0,3)), Point(e1.at<double>(0,4),e1.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
			line(image3, Point(e2.at<double>(0,2),e2.at<double>(0,3)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
			//S-S
			line(image3, Point(e1.at<double>(0,4),e1.at<double>(0,5)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(255, 255, 0), 5, 8,0);
			// S -H
			line(image3, Point((e2.at<double>(0,4) +e1.at<double>(0,4))/2,(e2.at<double>(0,5) +e1.at<double>(0,5))/2), Point(body_in.roi.x + body_in.roi.width/2.0,body_in.roi.y + body_in.roi.height/2.0), Scalar(255, 255,0), 5, 8,0);
		}
	}
}

// "euclidean" distance between blob (size and position)
double BlobTracker::blobDist(geometry_msgs::Point point, cv::KeyPoint keypoint)
{
	return sqrt(pow(point.x-keypoint.pt.x,2) + pow(point.y-keypoint.pt.y,2) + pow(point.z-keypoint.size,2));
}

// Extract blobs using face
cv::Mat BlobTracker::segmentFaces(cv::Mat input, body &face_in)
{
	cv::Mat image2(input);
//	cv::Mat image3 = image2.clone();
	cv::Mat image4;
	cvtColor(image2,image4,CV_BGR2HSV);
							
	vector<Mat> bgr_planes;
	split(image4, bgr_planes);
				
	MatND hist1, hist2;
	int histSize = 15;
	float h_range[] = {0, 180};
	float s_range[] = {0, 255};
	const float* rangesh = {h_range};
	const float* rangess = {s_range};
	cv::Rect rec_enlarged;
	rec_enlarged.x = face_in.roi.x+ face_in.roi.width/6;
	rec_enlarged.y = face_in.roi.y+ face_in.roi.height/6;
	rec_enlarged.width = face_in.roi.width - face_in.roi.width/6;
	rec_enlarged.height = face_in.roi.height- face_in.roi.height/6;
	cv::Mat subImg1 = bgr_planes[0](rec_enlarged);
	//~ medianBlur(subImg1,subImg1,7);
	cv::Mat subImg2 = bgr_planes[1](rec_enlarged);
	//~ medianBlur(subImg2,subImg2,7);
	calcHist(&subImg1, 1, 0, Mat(), hist1, 1, &histSize, &rangesh, true, false);
	calcHist(&subImg2, 1, 0, Mat(), hist2, 1, &histSize, &rangess, true, false);
	normalize(hist1, hist1, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(hist2, hist2, 0, 255, NORM_MINMAX, -1, Mat());
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	cv::Mat temp2(input.rows,input.cols,CV_64F);
	calcBackProject(&bgr_planes[0], 1, 0, hist1, temp1, &rangesh, 1, true);
	calcBackProject(&bgr_planes[1], 1, 0, hist2, temp2, &rangess, 1, true);
	
	bitwise_and(temp1,temp2,temp1,Mat());
	
	//~ // evaulate likelihood of each pixel being skin coloured
	//~ cv::Mat temp1(input.rows,input.cols,CV_64F);
	//~ //cv::Mat samples;
	//~ //image4.reshape(1,image4.rows*image4.cols).convertTo(samples,CV_32FC1,1.0/255.0);
	//~ cv:Mat probs;
	//~ for (int i = 0; i < input.rows; i++)
	//~ {
		//~ for (int k = 0; k < input.cols; k++)
		//~ {	
			//~ double p1 = 0, p2 = 0, p3 = 0;
			//~ for (int j = 0; j < (int)face_in.gmm_params1.size(); j++)
			//~ {
				//~ //ROS_WARN("%f %f %f",face_in.gmm_params1[bestIdx1].weight,face_in.gmm_params1[bestIdx1].mean,face_in.gmm_params1[bestIdx1].sigma);
				//~ //p1 = p1+face_in.gmm_params1[j].weight*1.0/(face_in.gmm_params1[j].sigma*sqrt(2*M_PI))*exp(-pow(std::min(bgr_planes[1].at<uchar>(i,k)-face_in.gmm_params1[j].mean,255-(bgr_planes[1].at<uchar>(i,k)-face_in.gmm_params1[j].mean)),2)/(2*pow(face_in.gmm_params1[j].sigma,2)));
				//~ //p2 = p2+face_in.gmm_params2[j].weight*1.0/(face_in.gmm_params2[j].sigma*sqrt(2*M_PI))*exp(-pow(std::min(bgr_planes[2].at<uchar>(i,k)-face_in.gmm_params2[j].mean,255-(bgr_planes[2].at<uchar>(i,k)-face_in.gmm_params2[j].mean)),2)/(2*pow(face_in.gmm_params2[j].sigma,2)));
				//~ //p3 = p3+face_in.gmm_params3[j].weight*1.0/(face_in.gmm_params3[j].sigma*sqrt(2*M_PI))*exp(-pow(std::min(bgr_planes[0].at<uchar>(i,k)-face_in.gmm_params3[j].mean,255-(bgr_planes[0].at<uchar>(i,k)-face_in.gmm_params3[j].mean)),2)/(2*pow(face_in.gmm_params3[j].sigma,2)));
				//~ p1 = p1+face_in.gmm_params1[j].weight*1.0/(face_in.gmm_params1[j].sigma*sqrt(2*M_PI))*exp(-pow(bgr_planes[1].at<uchar>(i,k)-face_in.gmm_params1[j].mean,2)/(2*pow(face_in.gmm_params1[j].sigma,2)));
				//~ p2 = p2+face_in.gmm_params2[j].weight*1.0/(face_in.gmm_params2[j].sigma*sqrt(2*M_PI))*exp(-pow(bgr_planes[2].at<uchar>(i,k)-face_in.gmm_params2[j].mean,2)/(2*pow(face_in.gmm_params2[j].sigma,2)));
				//~ //p3 = p3+face_in.gmm_params3[j].weight*1.0/(face_in.gmm_params3[j].sigma*sqrt(2*M_PI))*exp(-pow(bgr_planes[0].at<uchar>(i,k)-face_in.gmm_params3[j].mean,2)/(2*pow(face_in.gmm_params3[j].sigma,2)));
			//~ }
			//~ temp1.at<double>(i,k) = p1*p2;//*p3;
		//~ /*	if (gmm1->isTrained())
			//~ {
				//~ temp1.at<double>(i*input.rows + k) = exp(gmm1->predict(samples.row(i*input.rows + k),probs)[0]);
			//~ }
			//~ //ROS_INFO("%f ",exp(gmm1->predict(samples.row(i*input.rows + k),probs)[0]));*/
		//~ }
	//~ }
	//~ 
	//~ // normalise likelihood map
	//~ temp1 = temp1/cv::sum(temp1)[0];
	//~ double minVal,maxVal;
			//~ 
	//~ // find maxima and minima, scale for visualisation
	//~ minMaxLoc(temp1,&minVal,&maxVal,NULL,NULL,Mat());
	//~ ROS_INFO("minVal %f maxval %f",minVal,maxVal);
	//~ temp1.convertTo(temp1,CV_8UC1,255.0/maxVal,0);
	//~ 
	//~ // Smooth likelihood map
	//~ medianBlur(temp1,temp1,7);
	GaussianBlur(temp1,temp1,cv::Size(15,15),15,15,BORDER_DEFAULT);
	
	cvtColor(temp1,temp1,CV_GRAY2RGB);
		
	//~ cv::Mat image3 = image2.clone();
	cv::Mat image3 = temp1.clone(); // uncomment to view likelihood
	// Detect hands and update pose estimate
	blobDetector(temp1,face_in,image3);
				
	// Draw face rectangles
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(face_in.roi.x+ face_in.roi.width/5.0,face_in.roi.y+ face_in.roi.height/5.0), Point(face_in.roi.x+4*face_in.roi.width/5.0,face_in.roi.y+4*face_in.roi.height/5.0), Scalar(0,255,0), 4, 8, 0);
	
	return image3;
}

// Update list of faces
void BlobTracker::updateBlobFaces(const faceTracking::ROIArrayConstPtr& msg)
{
	
	if ((int)faces.size() == 0) // if no faces exist yet
	{
		for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list
		{
			body temp = body(cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height),5);
			temp.id = msg->ids[i];
			faces.push_back(temp); // add faces to list
		}
	}
	else
	{
		int count = (int)faces.size();
		for (int j = 0; j < count; j++)
		{
			faces[j].seen = false; //label all faces as unseen
		}
		for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list, iterate through detected faces
		{
			bool found = false;
			for (int j = 0; j < count; j++) // compare with existing faces
			{
				if (faces[j].id.compare(msg->ids[i]) == 0) // old face
				{
					faces[j].roi = cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height); //update roi
					faces[j].seen = true;
					found = true;
				}
			}
			if (!found) // new face
			{
				body temp = body(cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height),5);
				temp.id = msg->ids[i];
				faces.push_back(temp);
			}
		}
	}
}

// image and face roi callback
void BlobTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg)
{
	ROS_INFO ("Message received.");
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
		std::vector<cv::Mat> histImages;
		
		updateBlobFaces(msg); // update face list
		
		//~ int pixels = 100;
		cv::Mat bigImage = cv::Mat::zeros(image.rows,image.cols,image.type()); // large display image
		int posx = 0;
		int count = (int)faces.size();
		for (int i = 0; i < count; i++) // iterate through each face
		{	
			if (faces[i].seen) // track pose of existing faces
			{
				//~ cv::Mat image3;
				//~ cv::Rect tempRoi;
				//~ tempRoi.x = faces[i].roi.x + faces[i].roi.width/5.0;
				//~ tempRoi.y= faces[i].roi.y + faces[i].roi.height/5.0;
				//~ tempRoi.width = 4*faces[i].roi.width/5.0;
				//~ tempRoi.height = 4*faces[i].roi.height/5.0;
				//~ cvtColor(image(tempRoi),image3,CV_BGR2Lab);
							
				//~ vector<cv::Mat> bgr_planes;
				//~ blur(image3,image3,Size(15,15));
				//~ split(image3, bgr_planes);
				
				// update face skin colour model
//				cv::Mat obs;
	//			image3.reshape(1,image3.rows*image3.cols).convertTo(obs,CV_32FC1,1.0/255.0);
		//		gmm1->train(obs, likelihoods1, labels1, probs1);
			/*	model_params temp;
				std::vector<model_params> test;
				for (int k = 0; k < 5; k++) // skin colour model initial seed
				{
					temp.mean = k*255/(double)5;
					temp.sigma = 5;
					temp.weight = 1.0/(double)5;
					test.push_back(temp);
					
				}*/
				
				//~ faces[i].gmm_params1 = gmm_model->expectationMaximisation(faces[i].gmm_params1, bgr_planes[1]);
				//~ faces[i].gmm_params2 = gmm_model->expectationMaximisation(faces[i].gmm_params2, bgr_planes[2]);
			//	faces[i].gmm_params3 = gmm_model->expectationMaximisation(faces[i].gmm_params3, bgr_planes[0]);
				//faces[i].gmm_params1 = gmm_model->expectationMaximisation(test, bgr_planes[1]);
				//faces[i].gmm_params2 = gmm_model->expectationMaximisation(test, bgr_planes[2]);
				//faces[i].gmm_params3 = gmm_model->expectationMaximisation(test, bgr_planes[0]);
				// get histogram images to visualise skin colour model
				//~ histImages.push_back(getHistogram(image,tempRoi,faces[i].gmm_params1,faces[i].gmm_params2,faces[i].gmm_params3));
				if (i ==0)
				{
					Mat roiImgResult_top = bigImage(Rect(posx, 0, image.cols, image.rows));
					// extract faces and update pose
					Mat roiImg1 = segmentFaces(image,faces[i]);
					roiImg1.copyTo(roiImgResult_top); 
					//~ Mat roiImg2;
					//~ Mat roiImgResult_bottom = bigImage(Rect(posx,image.rows,image.cols,pixels)); 
					//~ resize(histImages.back(),roiImg2,Size(image.cols,pixels),0,0,INTER_LINEAR);
					//~ roiImg2.copyTo(roiImgResult_bottom);
					//~ posx = posx+image.cols;
				}
			}
			else // remove missing faces
			{
				count = count - 1;
				faces.erase(faces.begin() + i);
				i = i-1;
			}
		}		
		cv_bridge::CvImage img2;
		img2.encoding = "rgb8";
		img2.image = bigImage;			
		pub.publish(img2.toImageMsg());
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}

// Display pixel colour histogram and gmm approximation
//~ cv::Mat BlobTracker::getHistogram(cv::Mat input, cv::Rect roi, std::vector<model_params> gmm_params1, std::vector<model_params> gmm_params2, std::vector<model_params> gmm_params3)
//~ {
	//~ cv::Rect tempRoi;
	//~ tempRoi.x = roi.x + roi.width/5.0;
	//~ tempRoi.y = roi.y + roi.height/5.0;
	//~ tempRoi.width = 4*roi.width/5.0;
	//~ tempRoi.height = 4*roi.height/5.0;
	//~ cv::Mat image2(input, tempRoi);
	//~ cv::Mat image3 = image2.clone();
	//~ cvtColor(image3,image3,CV_BGR2Lab);
						//~ 
	//~ vector<Mat> bgr_planes;
	//~ split(image3, bgr_planes);
//~ 
	//~ int histSize = 255;
	//~ float range[] = {0, 255} ;
	//~ const float* histRange = {range};
				//~ 
	//~ Mat a_hist, b_hist;//, g_hist;
	//~ calcHist(&bgr_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, true, false);
	//~ calcHist(&bgr_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
	//~ //calcHist(&bgr_planes[0], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false);
	//~ int hist_w = 512; int hist_h = 400;
	//~ int bin_w = cvRound((double)hist_w/histSize);
//~ 
	//~ Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	//~ normalize(a_hist, a_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	//~ normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	//~ //normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		  //~ 
	//~ double f1[histSize], f2[histSize];//, f3[histSize];
	//~ double maxf1 = 0, maxf2 = 0;//, maxf3 = 0;
//~ 
	//~ for (int i = 0; i < histSize; i++)
	//~ {
		//~ f1[i] = 0;
		//~ f2[i] = 0;
		//~ //f3[i] = 0;
		//~ for (int k = 0; k < (int)gmm_params1.size(); k++)
		//~ {
			//~ //f1[i] = f1[i] + gmm_params1[k].weight*1/(gmm_params1[k].sigma*sqrt(2*M_PI))*exp(-pow(std::min(i-gmm_params1[k].mean,255-(i-gmm_params1[k].mean)),2)/(2*pow(gmm_params1[k].sigma,2)));
			//~ //f2[i] = f2[i] + gmm_params2[k].weight*1/(gmm_params2[k].sigma*sqrt(2*M_PI))*exp(-pow(std::min(i-gmm_params2[k].mean,255-(i-gmm_params2[k].mean)),2)/(2*pow(gmm_params2[k].sigma,2)));
			//~ //f3[i] = f3[i] + gmm_params3[k].weight*1/(gmm_params3[k].sigma*sqrt(2*M_PI))*exp(-pow(std::min(i-gmm_params3[k].mean,255-(i-gmm_params3[k].mean)),2)/(2*pow(gmm_params3[k].sigma,2)));
			//~ f1[i] = f1[i] + gmm_params1[k].weight*1/(gmm_params1[k].sigma*sqrt(2*M_PI))*exp(-pow(i-gmm_params1[k].mean,2)/(2*pow(gmm_params1[k].sigma,2)));
			//~ f2[i] = f2[i] + gmm_params2[k].weight*1/(gmm_params2[k].sigma*sqrt(2*M_PI))*exp(-pow(i-gmm_params2[k].mean,2)/(2*pow(gmm_params2[k].sigma,2)));
			//~ //f3[i] = f3[i] + gmm_params3[k].weight*1/(gmm_params3[k].sigma*sqrt(2*M_PI))*exp(-pow(i-gmm_params3[k].mean,2)/(2*pow(gmm_params3[k].sigma,2)));
		//~ }
		//~ if (f1[i] > maxf1)
		//~ {
			//~ maxf1 = f1[i];
		//~ }
				//~ 
		//~ if (f2[i] > maxf2)
		//~ {
			//~ maxf2 = f2[i];
		//~ }
		//~ 
		//~ //if (f3[i] > maxf3)
		//~ //{
			//~ //maxf3 = f3[i];
		//~ //}
	//~ }
			//~ 
	//~ f1[0] =f1[0]/maxf1*histImage.rows;
	//~ f2[0] =f2[0]/maxf2*histImage.rows;	
	//~ //	f3[0] =f3[0]/maxf3*histImage.rows;
	//~ for (int i = 1; i < histSize-1; i++)
	//~ {
		//~ f1[i] =f1[i]/maxf1*histImage.rows;
		//~ f2[i] =f2[i]/maxf2*histImage.rows;
		//~ //	f3[i] =f3[i]/maxf3*histImage.rows;
		//~ line(histImage, Point(bin_w*(i-1), hist_h - cvRound(a_hist.at<float>(i-1))), Point(bin_w*(i), hist_h - cvRound(a_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		//~ line(histImage, Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))), Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(0, 255, 255), 2, 8, 0);
//~ //		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1))), Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(255, 255, 255), 2, 8, 0);
		//~ line(histImage, Point(bin_w*(i-1), hist_h - cvRound(f1[i-1])), Point(bin_w*(i), hist_h - cvRound(f1[i])), Scalar(255, 0, 0), 2, 8, 0);
		//~ line(histImage, Point(bin_w*(i-1), hist_h - cvRound(f2[i-1])), Point(bin_w*(i), hist_h - cvRound(f2[i])), Scalar(0, 0, 255), 2, 8, 0);
//~ //		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(f3[i-1])), Point(bin_w*(i), hist_h - cvRound(f3[i])), Scalar(255, 0, 255), 2, 8, 0);
	//~ }
	//~ return histImage;
//~ }
