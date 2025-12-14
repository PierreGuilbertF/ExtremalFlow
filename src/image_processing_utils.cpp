#include "image_processing_utils.h"

namespace image_processing_utils {
void NormalizedImage(const cv::Mat &input_image, cv::Mat &normalized_image) {
  input_image.convertTo(normalized_image, CV_64F, 1.0 / 255.0);
}

void DenormalizedImage(const cv::Mat &normalized_image, cv::Mat &output_image) {
  normalized_image.convertTo(output_image, CV_8U, 255.0);
}

void Laplacian(const cv::Mat &input_image, cv::Mat &laplacian_image) {
  cv::Mat laplacian_kernel =
      (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
  cv::filter2D(input_image, laplacian_image, CV_64F, laplacian_kernel);
}
} // namespace image_processing_utils