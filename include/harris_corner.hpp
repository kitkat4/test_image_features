#pragma once

/*
  別にOpenCVの機能を使えばいいのだが、理解促進のために自作した
  Harrisコーナー検出器。
 */

#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>


class HarrisCorner{
public:

    HarrisCorner();
    ~HarrisCorner();

    void calcResponse(const cv::Mat& input_image,
                      cv::Mat& harris_response);
    
    static void nonMaximumSuppression(const cv::Mat& img_response,
                                      cv::Mat& img_binary_result,
                                      const double thresh,
                                      const int window_size);

private:

    void calcM(const cv::Mat& grad_x,
               const cv::Mat& grad_y,
               cv::Mat& M,
               const size_t row_idx,
               const size_t col_idx);

    static bool nonMaximumSuppressionCheckRow(const float val,
                                              float const * const row_head_pointer,
                                              const int window_center_col_idx,
                                              const int window_min,
                                              const int window_max,
                                              const int img_cols);

    
    double k_;
    int window_size_;        // must be an odd number

};

