FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/handBlobTracker/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_py"
  "../src/handBlobTracker/msg/__init__.py"
  "../src/handBlobTracker/msg/_face_hand.py"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_py.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
