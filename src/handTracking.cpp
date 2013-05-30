/* Body tracking using a particle filter and kinect upper body priors */
/* M. Burke*/
#include "handTracker.h"

int main( int argc, char** argv )
{
	ros::init(argc, argv, "blobTracking");
	
	HandTracker *tracker = new HandTracker();
	
	ros::spin();
	delete tracker;	
	return 0;
}
