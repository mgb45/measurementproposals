/* Body tracking using a particle filter and kinect upper body priors */
/* M. Burke*/
#include "blobTracker.h"

int main( int argc, char** argv )
{
	ros::init(argc, argv, "blobTracking");
	
	BlobTracker *tracker = new BlobTracker();
	
	ros::spin();
	delete tracker;	
	return 0;
}
