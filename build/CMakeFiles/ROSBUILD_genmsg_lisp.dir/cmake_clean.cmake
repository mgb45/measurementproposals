FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/handBlobTracker/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_lisp"
  "../msg_gen/lisp/HFPose2DArray.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_HFPose2DArray.lisp"
  "../msg_gen/lisp/HFPose2D.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_HFPose2D.lisp"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_lisp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
