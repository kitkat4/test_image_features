#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>

#include "my_utils_kk4.hpp"

#include <stdexcept>
#include <exception>
#include <string>
#include <iostream>

static int nms_window_size_ = 1;

bool nonMaximumSuppressionCheckRow(const float val,
                                   float const * const row_head_pointer,
                                   const int window_center_col_idx,
                                   const int window_min,
                                   const int window_max,
                                   const int img_cols){

    const int i_c = window_center_col_idx;

    bool skip = false;

    for(int j_c = window_min; j_c <= window_max; j_c++){

        if(i_c + j_c < 0){
            continue;
        }else if(i_c + j_c >= img_cols){
            continue;
        }

        float neigh_val = *(row_head_pointer + i_c + j_c);

        if(val < neigh_val){
            skip = true;
            break;
        }
    }

    return skip;
}

void nonMaximumSuppression(const cv::Mat& img_response,
                           cv::Mat& img_binary_result,
                           const double thresh,
                           const int window_size){

    if(img_response.type() != CV_32FC1){
        throw std::runtime_error("Invalid matrix type");
    }

    if(window_size % 2 == 0){
        throw std::runtime_error("window_size must be an odd number");
    }

    img_binary_result = cv::Mat::zeros(img_response.size(), CV_8UC1);

    const int window_min = - (window_size - 1) / 2;
    const int window_max = -window_min;

    for(int i_r = 0; i_r < img_response.rows; i_r++){
        
        float const * const row_data = img_response.ptr<float>(i_r);
        
        for(int i_c = 0; i_c < img_response.cols; i_c++){

            const float val = *(row_data + i_c);
            
            if(val < thresh){
                continue;
            }

            bool skip = false;

            // check the current row first
            skip = nonMaximumSuppressionCheckRow(val, row_data, i_c,
                                                 window_min, window_max,
                                                 img_response.cols);
            
            if(skip){
                continue;
            }

            // check the other rows
            for(int j_r = window_min; j_r <= window_max; j_r++){
                
                if(j_r == 0){
                    continue;
                }

                if(nonMaximumSuppressionCheckRow(
                       val, img_response.ptr<float>(i_r + j_r),
                       i_c, window_min, window_max, img_response.cols)){

                    skip = true;
                    break;
                }
            }

            if(skip){
                continue;
            }

            *(img_binary_result.ptr<uint8_t>(i_r) + i_c) = 255;
        }
    }

    return;
}


int main(int argc, char** argv){

    int camera_id = 0;
    int binarization_thresh = 10;
    int harris_k = 4;
    
    
    switch(argc){
        
    case 1:
        std::cout << "[ INFO] No camera ID specified. Default ID ("
                  << camera_id
                  << ") will be used." << std::endl;
        break;
        
    case 2:
        try{
            camera_id = std::stoi(argv[1]);
        }catch(const std::exception& e){
            std::cout << e.what() << std::endl;
            return 1;
        }
            
        break;
        
    default:
        std::cout << "Usage: "
                  << argv[0] << " <(optional) camera ID integer>" << std::endl;
        return 1;
    }

    const std::string window_name = argv[0];
    const std::string window_name_harris_response = "Harris corner response";

    cv::VideoCapture video(camera_id);

    if(video.isOpened()){
        std::cout << "[ INFO] Successfully opened video device "
                  << camera_id << std::endl;
    }else{
        std::cout << "[ERROR] Could not open video device "
                  << camera_id << std::endl;
        return 1;
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::namedWindow(window_name_harris_response,
                    cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
    cv::createTrackbar("harris_k (<set val> / 100)", window_name_harris_response,
                       &harris_k, 100, nullptr);
    cv::createTrackbar("binarization_thresh (<set val> / 1e6)", window_name_harris_response,
                       &binarization_thresh, 1000, nullptr);
    cv::createTrackbar("non maximum suppresion window size (<set val> * 2 + 1)",
                       window_name_harris_response,
                       &nms_window_size_, 15, nullptr);

    my_utils_kk4::Fps fps;
    my_utils_kk4::StopWatch fps_stop_watch;
    const double fps_show_interval = 1; // sec
    
    const char quit_key = 'q';
    std::cout << "[ INFO] Started main loop. Press "
              << quit_key << " to quit." << std::endl;

    cv::Mat image;
    cv::Mat gray_image;
    cv::Mat harris_response;
    cv::Mat harris_response_binary;

    fps_stop_watch.start();
    while(true){

        {  // calc and show FPS
            fps.trigger();
            if(fps_stop_watch.lap() > fps_show_interval){
                fps_stop_watch.stop();
                fps_stop_watch.reset();
                fps_stop_watch.start();
                std::cout << "\r[ INFO] " << fps.getFps() << " fps    " << std::flush;
            }
        }

        video >> image;

        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        {  // Harris corner detection
            harris_response = cv::Mat::zeros(image.size(), CV_32FC1);
            const int harris_block_size = 2;
            const int harris_aperture_size = 3;
            
            
            cv::cornerHarris(gray_image, harris_response, harris_block_size,
                             harris_aperture_size, harris_k / 100.0);
            // cv::threshold(harris_response, harris_response_binary,
            //               binarization_thresh / 10000.0, 255, cv::THRESH_BINARY);
            nonMaximumSuppression(harris_response, harris_response_binary,
                                  static_cast<double>(binarization_thresh) / 1e6,
                                  nms_window_size_ * 2 + 1);
        }

        cv::imshow(window_name, image);
        cv::imshow(window_name_harris_response, harris_response_binary);

        {  // Process key input
            const int key = cv::waitKey(1);

            if(key == quit_key){
                std::cout << std::endl
                          << "[ INFO] Quit" << std::endl;
                break;
            }
        }
    }

    return 0;
}
