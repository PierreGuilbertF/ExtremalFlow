#pragma once

#include <opencv2/opencv.hpp>

namespace image_processing_utils {
/**
 * @brief Normalize the image to the range [0, 1]
 * @param input_image The input image
 * @param normalized_image The normalized image
 */
void NormalizedImage(const cv::Mat &input_image, cv::Mat &normalized_image);

/**
 * @brief Denormalize the image to the range [0, 255]
 * @param normalized_image The normalized image
 * @param output_image The denormalized image
 */
void DenormalizedImage(const cv::Mat &normalized_image, cv::Mat &output_image);

/**
 * @brief Compute the Laplacian of the image
 * @param input_image The input image
 * @param laplacian_image The Laplacian image
 */
void Laplacian(const cv::Mat &input_image, cv::Mat &laplacian_image);

} // namespace image_processing_utils