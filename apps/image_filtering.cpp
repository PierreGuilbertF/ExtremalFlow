#include <iostream>
#include <opencv2/opencv.hpp>

#include "../src/euler_lagrange_flow.h"
#include "../src/image_processing_utils.h"

void ThikhonovFiltering(const cv::Mat &input, cv::Mat &output, double lambda) {
  // Normalized the image between [0, 1]
  cv::Mat image_normalized;
  image_processing_utils::NormalizedImage(input, image_normalized);

  cv::VideoWriter video_writer(
      "/Users/pierre.guilbert/dev/ExtremalFlow/results/output.avi",
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, input.size());

  cv::Mat u = image_normalized.clone();
  const double dt =
      100000.0 / (4.0 * lambda * static_cast<double>(u.rows * u.cols));
  std::cout << "dt: " << dt << std::endl;
  for (int k = 0; k < 600; ++k) {
    cv::Mat laplacian_image;
    image_processing_utils::Laplacian(u, laplacian_image);
    cv::Mat euler_lagrange =
        2.0 * (u - image_normalized) - lambda * laplacian_image;
    double grad_norm = cv::norm(euler_lagrange, cv::NORM_L2);
    std::cout << "grad_norm: " << grad_norm << std::endl;
    if (grad_norm < 1e-3)
      break;
    double total_energy = euler_lagrange.dot(euler_lagrange);
    u = u - dt * euler_lagrange;
    cv::min(cv::max(u, 0.0), 1.0, u);
    cv::Mat u_denormalized;
    image_processing_utils::DenormalizedImage(u, u_denormalized);
    video_writer.write(u_denormalized);
  }
  image_processing_utils::DenormalizedImage(u, output);
  video_writer.release();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string image_filename = argv[1];
  const cv::Mat image = cv::imread(image_filename);
  cv::Mat output_image;
  ThikhonovFiltering(image, output_image, 100.0);
  cv::imwrite("/Users/pierre.guilbert/dev/ExtremalFlow/results/input.jpg",
              image);
  cv::imwrite(argv[2], output_image);

  return EXIT_SUCCESS;
}