#include "blobTracker.h"

using namespace cv;
using namespace std;

BlobTracker::BlobTracker()
{
	namedWindow("win0", CV_WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow("win1", CV_WINDOW_AUTOSIZE );// Create a window for display.
	
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/blobImage",1); //ROS
	
	gmm_model = new GMM(255);
	im_received = false;
	
	image_sub.subscribe(nh, "/rgb/image_color", 1);
	roi_sub.subscribe(nh, "/faceROIs", 1);
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, roi_sub);
	sync->registerCallback(boost::bind(&BlobTracker::callback, this, _1, _2));
	
	params = new SimpleBlobDetector::Params;
	
    params->minArea = 400;
    params->maxArea = 10000;
    	
	params->filterByColor = false;
    params->filterByCircularity = false;
    params->filterByArea = true;
    params->filterByInertia = false;
    params->filterByConvexity = false;
}

BlobTracker::~BlobTracker()
{
	delete gmm_model;
	delete sync;
}

void BlobTracker::blobDetector(cv::Mat output, body &body_in, cv::Mat image3)
{
   
    double fmin = 100000, fmax = 0;
    for (int i = 0; i < (int)faces.size(); i++)
    {
		fmin = std::min(fmin,(double)std::min(faces[i].roi.height,faces[i].roi.width));
		fmax = std::max(fmax,(double)std::max(faces[i].roi.height,faces[i].roi.width));
	}
    params->minArea = M_PI*fmin*fmin/64.0;
    params->maxArea = M_PI*fmax*fmax/2.0;
    params->minDistBetweenBlobs = 5.0*fmin/6.0;

    params->minThreshold = 0;
    params->maxThreshold = 150;
    params->thresholdStep = 10;
    
    SimpleBlobDetector blobDetector(*params);
    blobDetector.create("SimpleBlob");
    
    //MSER mser_detector(5, M_PI*fmin*fmin/64.0, M_PI*fmax*fmax/2.0, .25, .2);
    //cvMSERParams( 5, 60, cvRound(.2*img->width*img->height), .25, .2 );
    vector<KeyPoint> keyPoints;
    blobDetector.detect(output, keyPoints);
    //mser_detector.detect(output, keyPoints);
    
    if ((int)keyPoints.size() >= 2)
    {
		int minIdx1 = -1;
		double minD1 = 100;
		int minIdx2 = -1; 
		double minD2 = 100;
		for (int i = 0; i < (int)keyPoints.size(); i++)
		{
			geometry_msgs::Point pt1;
			pt1.x = body_in.roi.x + body_in.roi.width/2.0;
			pt1.y = body_in.roi.y + body_in.roi.height/2.0;
			pt1.z = body_in.roi.height;
			if (blobDist(pt1,keyPoints[i]) > body_in.roi.width)
			{
				if (body_in.leftHand.z != -1)
				{
					double temp = blobDist(body_in.leftHand,keyPoints[i]);
					if (minD1 > temp)
					{
						minD1 = temp;
						minIdx1 = i;
					}
				}
				else
				{
					geometry_msgs::Point pt1;
					pt1.x = output.cols/4.0;
					pt1.y = output.rows/2.0;
					pt1.z = 20;
					double temp = blobDist(pt1,keyPoints[i]);
					if (minD1 > temp)
					{
						minD1 = temp;
						minIdx1 = i;
					}
				}
				
				if (body_in.rightHand.z != -1)
				{
					double temp = blobDist(body_in.rightHand,keyPoints[i]);
					if (minD2 > temp)
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
					double temp = blobDist(pt1,keyPoints[i]);
					if (minD2 > temp)
					{
						minD2 = temp;
						minIdx2 = i;
					}
				}
				ellipse(image3, keyPoints[i].pt, Size(keyPoints[i].size,keyPoints[i].size), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
			}
		}
		if ((minIdx1 != -1)&&(minIdx2 != -1)&&(minIdx1!=minIdx2))
		{
			cv::Mat measurement1(2,2,CV_64F);
			measurement1.at<double>(0,0) = body_in.roi.x + body_in.roi.width/2.0;
			measurement1.at<double>(0,1) = body_in.roi.y + body_in.roi.height/2.0;
			measurement1.at<double>(1,0) = keyPoints[minIdx1].pt.x;
			measurement1.at<double>(1,1) = keyPoints[minIdx1].pt.y;
			body_in.pf1->update(measurement1);
			
			cv::Mat measurement2(2,2,CV_64F);
			measurement2.at<double>(0,0) = body_in.roi.x + body_in.roi.width/2.0;
			measurement2.at<double>(0,1) = body_in.roi.y + body_in.roi.height/2.0;
			measurement2.at<double>(1,0) = keyPoints[minIdx2].pt.x;
			measurement2.at<double>(1,1) = keyPoints[minIdx2].pt.y;
			body_in.pf2->update(measurement2);
			
			cv::Mat e1 = body_in.pf1->getEstimator();
			cv::Mat e2 = body_in.pf2->getEstimator();
			
			body_in.leftHand.x = e1.at<double>(0,0);
			body_in.leftHand.y = e1.at<double>(0,1);
			body_in.leftHand.z = keyPoints[minIdx1].size;
			
			body_in.rightHand.x = e2.at<double>(0,0);
			body_in.rightHand.y = e2.at<double>(0,1);
			body_in.rightHand.z = keyPoints[minIdx2].size;
			
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
			//line(image3, Point((e2.at<double>(0,2) +e1.at<double>(0,2))/2,(e2.at<double>(0,3) +e1.at<double>(0,3))/2), Point(body_in.roi.x + body_in.roi.width/2.0,body_in.roi.y + body_in.roi.height/2.0), Scalar(255, 255, 0), 5, 8,0);
		}
	}
}

double BlobTracker::blobDist(geometry_msgs::Point point, cv::KeyPoint keypoint)
{
	return sqrt(pow(point.x-keypoint.pt.x,2) + pow(point.y-keypoint.pt.y,2) + pow(point.z-keypoint.size,2));
}


cv::Mat BlobTracker::segmentFaces(cv::Mat input, body &face_in)
{
	cv::Mat image2(input);
	cv::Mat image3 = image2.clone();
	cv::Mat image4;
	cvtColor(image3,image4,CV_BGR2Lab);
							
	vector<Mat> bgr_planes;
	split(image4, bgr_planes);
				
	int bestIdx1 = 0, bestIdx2 = 0;
	double tempMax1=0, tempMax2=0;
	for (int i = 0; i < (int)face_in.gmm_params1.size(); i++)
	{
		if (face_in.gmm_params1[i].weight > tempMax1)
		{
			tempMax1 = face_in.gmm_params1[i].weight;
			bestIdx1 = i;
		}
		if (face_in.gmm_params2[i].weight > tempMax2)
		{
			tempMax2 = face_in.gmm_params2[i].weight;
					bestIdx2 = i;
		}
	}
			
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	for (int i = 0; i < input.rows; i++)
	{
		for (int k = 0; k < input.cols; k++)
		{	
			double p1 = 0;
			p1 = 1.0/(face_in.gmm_params1[bestIdx1].sigma*sqrt(2*M_PI))*exp(-pow(std::min(bgr_planes[1].at<uchar>(i,k)-face_in.gmm_params1[bestIdx1].mean,255-(bgr_planes[1].at<uchar>(i,k)-face_in.gmm_params1[bestIdx1].mean)),2)/(2*pow(face_in.gmm_params1[bestIdx1].sigma,2)));
			p1 = p1*1.0/(face_in.gmm_params2[bestIdx2].sigma*sqrt(2*M_PI))*exp(-pow(std::min(bgr_planes[2].at<uchar>(i,k)-face_in.gmm_params2[bestIdx2].mean,255-(bgr_planes[2].at<uchar>(i,k)-face_in.gmm_params2[bestIdx2].mean)),2)/(2*pow(face_in.gmm_params2[bestIdx2].sigma,2)));
			temp1.at<double>(i,k) = p1;
		}
	}
	
	
			
	temp1 = temp1/cv::sum(temp1)[0];
	double minVal,maxVal;
			
	minMaxLoc(temp1,&minVal,&maxVal,NULL,NULL,Mat());
	temp1.convertTo(temp1,CV_8UC1,255.0/maxVal,0);
	
	medianBlur(temp1,temp1,7);
	/*adaptiveThreshold(temp1,temp1,255,ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,3,0);
	int erosion_type = MORPH_ELLIPSE;
    int erosion_size = 5;
    Mat element = getStructuringElement(erosion_type, Size(2*erosion_size + 1, 2*erosion_size+1), Point(erosion_size, erosion_size));
    dilate(temp1,temp1,element);
    erode(temp1,temp1,element);*/

	cvtColor(temp1,temp1,CV_GRAY2RGB);
	blobDetector(temp1,face_in,image3);
				
	rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
	rectangle(image3, Point(face_in.roi.x+ face_in.roi.width/5.0,face_in.roi.y+ face_in.roi.height/5.0), Point(face_in.roi.x+4*face_in.roi.width/5.0,face_in.roi.y+4*face_in.roi.height/5.0), Scalar(0,255,0), 4, 8, 0);
	
	return image3;
}

void BlobTracker::updateBlobFaces(const faceTracking::ROIArrayConstPtr& msg)
{
	if ((int)faces.size() == 0)
	{
		for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list
		{
			body temp = body(cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height),5);
			temp.id = msg->ids[i];
			faces.push_back(temp);
		}
	}
	else
	{
		int count = (int)faces.size();
		for (int j = 0; j < count; j++)
		{
			faces[j].seen = false;
		}
		for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list
		{
			bool found = false;
			for (int j = 0; j < count; j++)
			{
				if (faces[j].id.compare(msg->ids[i]) == 0) // old face
				{
					faces[j].roi = cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height);
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

void BlobTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg)
{
	ROS_INFO ("Message received.");
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
		std::vector<cv::Mat> histImages;
		
		updateBlobFaces(msg);
		
		int pixels = 100;
		cv::Mat bigImage = cv::Mat::zeros(image.rows + pixels,image.cols*(int)msg->ROIs.size(),image.type());
		int posx = 0;
		int count = (int)faces.size();
		for (int i = 0; i < count; i++)
		{	
			if (faces[i].seen)
			{
				cv::Mat image3;
				cv::Rect tempRoi;
				tempRoi.x = faces[i].roi.x + faces[i].roi.width/5.0;
				tempRoi.y= faces[i].roi.y + faces[i].roi.height/5.0;
				tempRoi.width = 4*faces[i].roi.width/5.0;
				tempRoi.height = 4*faces[i].roi.height/5.0;
				cvtColor(image(tempRoi),image3,CV_BGR2Lab);
							
				vector<cv::Mat> bgr_planes;
				blur(image3,image3,Size(15,15));
				split(image3, bgr_planes);
				
				faces[i].gmm_params1 = gmm_model->expectationMaximisation(faces[i].gmm_params1, bgr_planes[1]);
				faces[i].gmm_params2 = gmm_model->expectationMaximisation(faces[i].gmm_params2, bgr_planes[2]);
				histImages.push_back(getHistogram(image,tempRoi,faces[i].gmm_params1,faces[i].gmm_params2));

				
				Mat roiImgResult_top = bigImage(Rect(posx, 0, image.cols, image.rows));
				Mat roiImg1 = segmentFaces(image,faces[i]);
				roiImg1.copyTo(roiImgResult_top); 
				Mat roiImg2;
				Mat roiImgResult_bottom = bigImage(Rect(posx,image.rows,image.cols,pixels)); 
				resize(histImages.back(),roiImg2,Size(image.cols,pixels),0,0,INTER_LINEAR);
				roiImg2.copyTo(roiImgResult_bottom);
				posx = posx+image.cols;
			}
			else
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

cv::Mat BlobTracker::getHistogram(cv::Mat input, cv::Rect roi, std::vector<model_params> gmm_params1, std::vector<model_params> gmm_params2)
{
	cv::Rect tempRoi;
	tempRoi.x = roi.x + roi.width/5.0;
	tempRoi.y = roi.y + roi.height/5.0;
	tempRoi.width = 4*roi.width/5.0;
	tempRoi.height = 4*roi.height/5.0;
	cv::Mat image2(input, tempRoi);
	cv::Mat image3 = image2.clone();
	cvtColor(image3,image3,CV_BGR2Lab);
						
	vector<Mat> bgr_planes;
	split(image3, bgr_planes);

	int histSize = 255;
	float range[] = {0, 255} ;
	const float* histRange = {range};
				
	Mat b_hist, r_hist;
	calcHist(&bgr_planes[1], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false);
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w/histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		  
	double f1[histSize], f2[histSize];
	double maxf1 = 0, maxf2 = 0;
	for (int i = 0; i < histSize; i++)
	{
		f1[i] = 0;
		f2[i] = 0;
		for (int k = 0; k < (int)gmm_params1.size(); k++)
		{
			f1[i] = f1[i] + gmm_params1[k].weight*1/(gmm_params1[k].sigma*sqrt(2*M_PI))*exp(-pow(std::min(i-gmm_params1[k].mean,255-(i-gmm_params1[k].mean)),2)/(2*pow(gmm_params1[k].sigma,2)));
			f2[i] = f2[i] + gmm_params2[k].weight*1/(gmm_params2[k].sigma*sqrt(2*M_PI))*exp(-pow(std::min(i-gmm_params2[k].mean,255-(i-gmm_params1[k].mean)),2)/(2*pow(gmm_params2[k].sigma,2)));
		}
		if (f1[i] > maxf1)
		{
			maxf1 = f1[i];
		}
				
		if (f2[i] > maxf2)
		{
			maxf2 = f2[i];
		}
	}
	for (int i = 0; i < histSize; i++)
	{
		f1[i] =f1[i]/maxf1*histImage.rows;
		f2[i] =f2[i]/maxf2*histImage.rows;
	}
			
	for (int i = 1; i < histSize-1; i++)
	{
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))), Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1))), Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 255, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(f1[i-1])), Point(bin_w*(i), hist_h - cvRound(f1[i])), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(f2[i-1])), Point(bin_w*(i), hist_h - cvRound(f2[i])), Scalar(0, 0, 255), 2, 8, 0);
	}
	return histImage;
}
