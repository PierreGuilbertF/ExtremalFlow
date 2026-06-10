#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include "riemanian_geometry.h"

class Parameterization {
public:
    Eigen::Matrix<double, Eigen::Dynamic, 1> operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> & x) const {
        Eigen::Vector3d y;
        y(0) = cos(2 * M_PI * x(0)) * cos(M_PI * x(1) - M_PI / 2.0);
        y(1) = sin(2 * M_PI * x(0)) * cos(M_PI * x(1) - M_PI / 2.0);
        y(2) = sin(M_PI * x(1) - M_PI / 2.0);

        auto n = y / y.norm();
        y += 0.5 * (x(0) * x(0) + x(1) * x(1)) * n;
        return y;
    }

    int GetInputDim() const { return input_dim; }
    int GetOutputDim() const { return output_dim; }

protected:
    int input_dim = 2;
    int output_dim = 3;
};

class Sphere {
public:
    Eigen::Matrix<double, Eigen::Dynamic, 1> operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> & x) const {
        Eigen::Vector3d y;
        y(0) = cos(2 * M_PI * x(0)) * cos(M_PI * x(1) - M_PI / 2.0);
        y(1) = sin(2 * M_PI * x(0)) * cos(M_PI * x(1) - M_PI / 2.0);
        y(2) = sin(M_PI * x(1) - M_PI / 2.0);
        return y;
    }

    int GetInputDim() const { return input_dim; }
    int GetOutputDim() const { return output_dim; }

protected:
    int input_dim = 2;
    int output_dim = 3;
};

int main(int argc, char** argv)
{
    Sphere f;
    std::ofstream file("/Users/pierre.guilbert/Downloads/visu/manifold.csv");
    file << "x,y,z,curvature" << std::endl;
    for (double x = 0; x <= 1.0; x+= 0.01) {
        for (double y = 0; y <= 1.0; y+= 0.01) {
            const Eigen::Vector2d uv(x, y);
            const Eigen::Vector3d xyz = f(uv);
            // Matrix containing local basis of tangent space
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J = riemanian_geometry::GetJacobean(f, uv);
            // Matrix containing local metric tensor
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = J.transpose() * J;
            // Get Hessian
            const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> hessian = riemanian_geometry::GetHessian(f, uv);
            // Get normals
            const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> normals = riemanian_geometry::GetNormals(J);
            // Get Christoffels
            const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> christoffels = riemanian_geometry::GetChristoffel(f, J, hessian, uv);

            const auto n = normals[0];
            Eigen::Matrix<double, 2, 2> II;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    II(i, j) = hessian[i + j * 2].dot(n);
                }
            }
            Eigen::Matrix<double, 2, 2> W = G.inverse() * II;
            const double curvature = W.determinant();
            std::cout << "Curvature: " << curvature << std::endl;
            file << xyz(0) << "," << xyz(1) << "," << xyz(2) << ", " << curvature << std::endl;
        }
    }
    return 0;
}
