cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(test_image_features)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include
  ${OpenCV_INCLUDE_DIR}
  ${Eigen3_INCLUDE_DIR}
)

add_executable(test_harris_corner_with_camera
  src/test_harris_corner_with_camera_main.cpp
)

target_link_libraries(test_harris_corner_with_camera
  ${OpenCV_LIBS}
  ${Eigen3_LIBS}
)




