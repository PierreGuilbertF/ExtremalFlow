#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace riemanian_geometry {
const double kEpsilon = 1e-7;
const double kEpsilon2 = 1e-6;

template <typename F>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GetJacobean(const F & f, const Eigen::Matrix<double, Eigen::Dynamic, 1> & x) {
    const int dim_in = f.GetInputDim();
    const int dim_out = f.GetOutputDim();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobean(dim_out, dim_in);
    for (int j = 0; j < dim_in; ++j) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> dx = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(dim_in, 1);
        dx(j) += kEpsilon;
        const Eigen::Matrix<double, Eigen::Dynamic, 1> partial_diff = (f(x + dx) - f(x - dx)) / (2.0 * kEpsilon);
        jacobean.col(j) = partial_diff;
    }
    return jacobean;
}

// Return the Hessian "tensor" of a function from R^n to R^m. The result is a nxnxm tensor.
template <typename F>
std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> GetHessian(const F & f, const Eigen::Matrix<double, Eigen::Dynamic, 1> & x) {
    const int dim_in = f.GetInputDim();
    const int dim_out = f.GetOutputDim();

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> hessian(dim_in * dim_in);
    for (int i = 0; i < dim_in; ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> dx1 = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(dim_in, 1);
        dx1(i) += kEpsilon2;
        for (int j = 0; j < dim_in; ++j) {
          Eigen::Matrix<double, Eigen::Dynamic, 1> dx2 = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(dim_in, 1);
          dx2(j) += kEpsilon2;
          const Eigen::Matrix<double, Eigen::Dynamic, 1> y = (f(x + dx1 + dx2) + f(x -dx1 - dx2) - f(x - dx1 + dx2) - f(x+dx1 - dx2)) / (4.0 * kEpsilon2 * kEpsilon2);
          hessian[i + dim_in * j] = y;
        }
    }
    return hessian;
}

// Return an orthgonal basis of the orthogonal of span(X(1), ..., X(n)).
std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> GetNormals(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & X) {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> svd(X,Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto U = svd.matrixU();
    const auto singular_values = svd.singularValues();

    int n = X.rows();
    int k = X.cols();

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> normals;
    int rank = 0;
    for (int i = 0; i < singular_values.size(); ++i)
    {
        if (singular_values(i) > kEpsilon) rank++;
    }

    for (int i = rank; i < n; ++i)
    {
        normals.push_back(U.col(i));
    }
    return normals;
}

// Return the Christoffel symbol for point x, jacobean J, hessian H and parameterization f
template <typename F>
std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> GetChristoffel(const F & f, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & J, const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> & H, const Eigen::Matrix<double, Eigen::Dynamic, 1> & x) {
    const int dim_in = f.GetInputDim();

    const auto G_inv = (J.transpose() * J).inverse();
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> christoffels(dim_in * dim_in);
    for (int i = 0; i < dim_in; ++i) {
        for (int j = 0; j < dim_in; ++j) {
            christoffels[i + dim_in * j] = G_inv * J.transpose() * H[i + dim_in * j];
        }
    }

    return christoffels;
}

} // namespace riemanian_geometry
