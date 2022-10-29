#include <harris_corner.hpp>


HarrisCorner::HarrisCorner(){

}

HarrisCorner::~HarrisCorner(){

}

void HarrisCorner::calcResponse(const cv::Mat& input_image,
                                cv::Mat& harris_response){

    assert(input_image.type() == CV_32FC1);

    cv::Mat result = cv::Mat::zeros(input_image.size(), CV_32FC1);

    cv::Mat M = cv::Mat::zeros(cv::Size(2, 2), CV_32FC1);

    cv::Mat grad_x, grad_y;

    cv::Sobel(input_image, grad_x, CV_32F, 1, 0, 1);
    cv::Sobel(input_image, grad_y, CV_32F, 0, 1, 1);

    for(size_t row_idx = 0; row_idx < input_image.rows; row_idx++){

        for(size_t col_idx = 0; col_idx < input_image.cols; col_idx++){

            calcM(grad_x, grad_y, M, row_idx, col_idx);
        }
    }

    
}

void HarrisCorner::nonMaximumSuppression(const cv::Mat& img_response,
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


void HarrisCorner::calcM(const cv::Mat& grad_x,
                         const cv::Mat& grad_y,
                         cv::Mat& M,
                         const size_t row_idx,
                         const size_t col_idx){

    // w_ represents window
    
    const int half_w_size = (window_size_ - 1) / 2;

    
    for(int tmp_row_idx = static_cast<int>(row_idx) - half_w_size;
        tmp_row_idx <= row_idx + half_w_size;
        tmp_row_idx++){

        if(tmp_row_idx < 0 || tmp_row_idx >= grad_x.rows){
            continue;
        }

        for(int tmp_col_idx = static_cast<int>(col_idx) - half_w_size;
            tmp_col_idx <= col_idx + half_w_size;
            tmp_col_idx++){

            if(tmp_col_idx < 0 || tmp_col_idx >= grad_x.cols){
                continue;
            }

            
        }
    }
}

bool HarrisCorner::nonMaximumSuppressionCheckRow(const float val,
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






