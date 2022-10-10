#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>

#include "my_utils_kk4.hpp"

#include <exception>
#include <string>
#include <iostream>


int main(int argc, char** argv){

    int camera_id = 0;
    
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
    cv::namedWindow(window_name_harris_response, cv::WINDOW_NORMAL);
    

    my_utils_kk4::Fps fps;
    my_utils_kk4::StopWatch fps_stop_watch;
    const double fps_show_interval = 1; // sec
    
    const char quit_key = 'q';
    std::cout << "[ INFO] Started main loop. Press "
              << quit_key << " to quit." << std::endl;

    cv::Mat image;
    cv::Mat gray_image;
    cv::Mat harris_response;
    cv::Mat harris_response_to_display;

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
            const double harris_k = 0.04;
            cv::cornerHarris(gray_image, harris_response, harris_block_size,
                             harris_aperture_size, harris_k);
            cv::normalize(harris_response, harris_response_to_display,
                          0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::convertScaleAbs(harris_response_to_display,
                                harris_response_to_display);
        }

        cv::imshow(window_name, image);
        cv::imshow(window_name_harris_response, harris_response_to_display);

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
