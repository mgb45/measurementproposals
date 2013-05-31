FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/handBlobTracker/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_cpp"
  "../msg_gen/cpp/include/handBlobTracker/HFPose2DArray.h"
  "../msg_gen/cpp/include/handBlobTracker/HFPose2D.h"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_cpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
