#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

const int kMaxIteration = 10000;
const int kNumPoints = 100;
const double kRadius = 950;
const Eigen::Vector2d kCenter(1000, 1000);
const double w0 = 5.0;
const double w1 = 1.0;
const double w2 = 0.1;


using Contour = std::vector<Eigen::Vector2d>;

struct ImageDerivative{
    cv::Mat Ix;
    cv::Mat Iy;
    cv::Mat Ixx;
    cv::Mat Iyy;
    cv::Mat Ixy;
};

double ContourNorm(const Contour & contour) {
    double norm = 0;
    for (const auto & val : contour) {
        norm += val(0) * val(0) + val(1) * val(1);
    }
    return std::sqrt(norm);
}

void ComputeContourDerivative(const Contour & contour, Contour & derivative_contour) {
    int n = contour.size();
    const double h = 2.0 * M_PI / static_cast<double>(kNumPoints);
    for (int k = 0; k < n; ++k) {
        const int k_prev = (k - 1 + n) % n;
        const int k_next = (k + 1) % n;
        derivative_contour[k] = (contour[k_next] - contour[k_prev]) / (2.0 * h);
    }
}

void ComputeEulerLagrange(const ImageDerivative & Ig, const Contour & contour, Contour & euler_lag) {
  Contour derivative(contour.size());
  Contour derivative_2(contour.size());
  Contour derivative_3(contour.size());
  Contour derivative_4(contour.size());
  ComputeContourDerivative(contour, derivative);
  ComputeContourDerivative(derivative, derivative_2);
  ComputeContourDerivative(derivative_2, derivative_3);
  ComputeContourDerivative(derivative_3, derivative_4);
  for (int k = 0; k < contour.size(); ++k) {
    // Term associated to image gradient energy
    // With nearest neighbor interpolation for now
    const int x = static_cast<int>(std::round(contour[k](0)));
    const int y = static_cast<int>(std::round(contour[k](1)));
    Eigen::Matrix<double, 2, 2> H;
    Eigen::Vector2d G;
    H(0, 0) = Ig.Ixx.at<double>(y, x);
    H(1, 1) = Ig.Iyy.at<double>(y, x);
    H(0, 1) = Ig.Ixy.at<double>(y, x);
    H(1, 0) = H(0, 1);
    G(0) = Ig.Ix.at<double>(y, x);
    G(1) = Ig.Iy.at<double>(y, x);
    // Length reduction term:
    const Eigen::Vector2d length_reduction_term = -2.0 * derivative_2[k];
    // we use the potential 1 / (1 + ||\nabla I||^2) so the derivative:
    const Eigen::Vector2d image_gradient_term = -2.0 * H * G / std::pow(1 + G(0) * G(0) + G(1) * G(1), 2);
    // Area reduction term:
    const Eigen::Vector2d area_reduction_term(derivative[k](1), -derivative[k](0));
    euler_lag[k] = w0 * image_gradient_term + w1 * area_reduction_term + 2.0 * w2 * derivative_4[k];
  }
}

void UpdateContourWithGradientFlow(Contour & contour, const Contour & euler_lag, double eta = 1e-4) {
    for (int k = 0; k < contour.size(); ++k) {
        contour[k] -= eta * euler_lag[k];
    }
}

void DrawContour(cv::Mat & img, const std::vector<Eigen::Vector2d> & contour) {
    for (int k = 0; k < contour.size(); ++k) {
        const int k_next = (k + 1) % static_cast<int>(contour.size());
        const cv::Point2d p1(contour[k](0), contour[k](1));
        const cv::Point2d p2(contour[k_next](0), contour[k_next](1));
        cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1);
    }
    for (int k = 0; k < contour.size(); ++k) {
        const cv::Point2d p(contour[k](0), contour[k](1));
        cv::circle(img, p, 2, cv::Scalar(255, 0, 0), -1);
    }
    return;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string image_filename = argv[1];
  const cv::Mat image = cv::imread(image_filename);

  // Initialize circle contour
  Contour contour(kNumPoints);
  for (int k = 0; k < kNumPoints; ++k) {
    const double s = static_cast<double>(k) / static_cast<double>(kNumPoints);
    const double angle = s * 2.0 * M_PI;
    contour[k] = Eigen::Vector2d(kRadius * std::cos(angle), kRadius * std::sin(angle)) + kCenter;
  }

  const std::string output_filename(argv[2]);
  cv::VideoWriter video_writer(output_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, image.size());
  
  // Pre-compute Gradient and Hessian of the image
  cv::Mat grey;
  cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
  grey.convertTo(grey, CV_64F);
  cv::GaussianBlur(grey, grey, cv::Size(0, 0), 7, 7, cv::BORDER_REFLECT);

  ImageDerivative Ig;
  // Grad
  cv::Sobel(grey, Ig.Ix, CV_64F, 1, 0, 3); // d/dx
  cv::Sobel(grey, Ig.Iy, CV_64F, 0, 1, 3); // d/dy
  // Hessian
  cv::Sobel(Ig.Ix, Ig.Ixx, CV_64F, 1, 0, 3);
  cv::Sobel(Ig.Ix, Ig.Ixy, CV_64F, 0, 1, 3);
  cv::Sobel(Ig.Iy, Ig.Iyy, CV_64F, 0, 1, 3);

  std::vector<Eigen::Vector2d> euler_lag(contour.size());
  for (int iteration = 0; iteration < kMaxIteration; iteration++) {
    std::cout << "Iteration: " << iteration << " / " << kMaxIteration << std::endl;
    ComputeEulerLagrange(Ig, contour, euler_lag);
    UpdateContourWithGradientFlow(contour, euler_lag);
    const double euler_lag_norm = ContourNorm(euler_lag);
    std::cout << "Euler-Lagrange norm: " << euler_lag_norm << std::endl;
    if (iteration % 50 == 0) {
        cv::Mat temp_image;
        image.copyTo(temp_image);
        DrawContour(temp_image, contour);
        video_writer.write(temp_image);
    }
  }

  video_writer.release();

  return EXIT_SUCCESS;
}