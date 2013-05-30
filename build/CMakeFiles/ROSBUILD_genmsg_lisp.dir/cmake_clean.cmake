FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/handBlobTracker/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_lisp"
  "../msg_gen/lisp/face_hand.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_face_hand.lisp"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_lisp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
