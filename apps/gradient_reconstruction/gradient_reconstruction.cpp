#include <iostream>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

struct ImageGradient {
  cv::Mat Rx, Ry;
  cv::Mat Gx, Gy;
  cv::Mat Bx, By;
};

void ComputeImageGradient(const cv::Mat & image, ImageGradient & Ig) {
  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);

  cv::Sobel(channels[0], Ig.Rx, CV_64F, 1, 0, 3);
  cv::Sobel(channels[0], Ig.Ry, CV_64F, 0, 1, 3);

  cv::Sobel(channels[1], Ig.Gx, CV_64F, 1, 0, 3);
  cv::Sobel(channels[1], Ig.Gy, CV_64F, 0, 1, 3);

  cv::Sobel(channels[2], Ig.Bx, CV_64F, 1, 0, 3);
  cv::Sobel(channels[2], Ig.By, CV_64F, 0, 1, 3);
}

void ReconstructGradient(const cv::Mat & gradient_x, const cv::Mat & gradient_y, cv::Mat & reconstruction) {
  // We will look for the scalar field u that satisfies:
  // min(\int ||\nabla u - X||^2).
  // The Euler-Lagrangian is 2(\Delta u - div(X))
  // Thus, u satisfies \Delta u = div(X)
  std::vector<Eigen::Triplet<float>> non_zero_element;
  const int H = gradient_x.size().height;
  const int W = gradient_x.size().width;
  Eigen::VectorXf Values = Eigen::VectorXf(H * W);
  std::cout << "Reconstructing from gradient with dimension: " << H << "x" << W << std::endl;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        if (x == 0 && y == 0) {
            non_zero_element.emplace_back(0, 0, 1.0f);
            Values(0) = 0.0f;
            continue;
        }
        float div = 0.0f;
        if (x > 0) div += static_cast<float>(gradient_x.at<double>(y,x) - gradient_x.at<double>(y,x-1));
        if (y > 0) div += static_cast<float>(gradient_y.at<double>(y,x) - gradient_y.at<double>(y-1,x));

        // Expected value
        const int id0 = x + y * W;
        Values(id0) = div;

        // middle point
        non_zero_element.emplace_back(id0, id0, -4.0);
        // 4 corners
        if (x > 0)     non_zero_element.emplace_back(id0, x - 1 + y * W, 1.0f);
        if (x < W-1)   non_zero_element.emplace_back(id0, x + 1 + y * W, 1.0f);
        if (y > 0)     non_zero_element.emplace_back(id0, x + (y - 1) * W, 1.0f);
        if (y < H-1)   non_zero_element.emplace_back(id0, x + (y + 1) * W, 1.0f);
    }
  }

  std::cout << "Creating sparse matrix..." << std::endl;
  Eigen::SparseMatrix<float> A(H * W, H * W);
  A.setFromTriplets(non_zero_element.begin(), non_zero_element.end());

  //Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<float>> solver;
  //solver.compute(A);
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> chol(
                A);  // performs a Cholesky factorization of A
  std::cout << "Solving the problem..." << std::endl;
  const Eigen::VectorXf X = chol.solve(Values);
  std::cout << "Problem solved" << std::endl;

  reconstruction = cv::Mat::zeros(gradient_x.size(), CV_64F);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        const int id0 = x + y * W;
        reconstruction.at<double>(y, x) = static_cast<double>(X(id0));
    }
  }
}


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <input_image> <output_image>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string image_filename = argv[1];
  cv::Mat image = cv::imread(image_filename);
  image.convertTo(image, CV_64FC3);
  cv::resize(image, image, cv::Size(1000,1000), cv::INTER_LINEAR);

  const std::string image_filename2 = argv[2];
  cv::Mat image2 = cv::imread(image_filename2);
  image2.convertTo(image2, CV_64FC3);
  cv::resize(image2, image2, cv::Size(100,100), cv::INTER_LINEAR);
  
  cv::Rect roi(500, 500, image2.cols, image2.rows);
  image2.copyTo(image(roi));
  cv::imwrite("brutal_overlay.png", image);

  ImageGradient Ig1, Ig2;
  ComputeImageGradient(image, Ig1);
  ComputeImageGradient(image2, Ig2);

  // Incrust I2 gradient in I1
  const int x = 500;
  const int y = 500;

  Ig2.Rx.copyTo(Ig1.Rx(roi));
  Ig2.Ry.copyTo(Ig1.Ry(roi));

  Ig2.Gx.copyTo(Ig1.Gx(roi));
  Ig2.Gy.copyTo(Ig1.Gy(roi));

  Ig2.Bx.copyTo(Ig1.Bx(roi));
  Ig2.By.copyTo(Ig1.By(roi));
  

  cv::Mat R, G, B;
  ReconstructGradient(Ig1.Rx, Ig1.Ry, R);
  ReconstructGradient(Ig1.Gx, Ig1.Gy, G);
  ReconstructGradient(Ig1.Bx, Ig1.By, B);

  cv::Mat rI;
  std::vector<cv::Mat> channels = {R, G, B};
  cv::merge(channels, rI);

  cv::normalize(rI, rI, 0, 255, cv::NORM_MINMAX);
  rI.convertTo(rI, CV_8UC3);


  cv::imwrite("gradient_x.png", rI);

  return EXIT_SUCCESS;
}