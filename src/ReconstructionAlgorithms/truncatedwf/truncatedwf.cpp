#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>

namespace py = pybind11;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

double truncInitFilter(double y_i, double lambda_0_sq, double alpha_f_sq) {
    if(std::abs(y_i) <= lambda_0_sq * alpha_f_sq) {
        return y_i;
    } 
    return 0;
}

Eigen::VectorXd truncSpectralInit(Eigen::VectorXd y, RowMatrixXd A, double alpha_f) {
    int n = A.cols();
    int m = A.rows();
    double alpha_f_sq = std::pow(alpha_f, 2);

    double lamba_0_sq = y.sum() / m;
    Eigen::VectorXd condY = y.unaryexpr(std::ptr_fun(truncInitFilter));

    RowMatrixXd Y = RowMatrixXd::Zero(n ,n);

    return y; //Placeholder.
}

bool inE1(Eigen::VectorXd f, Eigen::VectorXd a_i, double alpha_lb, double alpha_ub, int n) {
    double v1 = (std::sqrt(n) / (std::sqrt(a_i.dot(a_i)))) * (std::abs(a_i.dot(f)) / std::sqrt(f.dot(f)));
    return (alpha_lb <= v1) && (v1 <= alpha_ub);
}

double calcKt(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, int m) {
    //This assumes real phase retrieval, i.e. input vectors contain real values.
    double sum = 0;
    for(int l = 0; l < m; l++) {
        sum += std::abs(y[l] - std::pow(std::abs(A.row(l).dot(f)), 2));
    }
    return sum / m;
}

bool inE2(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, double alpha_f, int n, int m, int i) {
    Eigen::VectorXd a_i = A.row(i);

    double lhs = std::abs(y[i] - std::pow(a_i.dot(f), 2));

    double rhs = alpha_f * calcKt(f, y, A, m) * (std::sqrt(n) / (std::sqrt(a_i.dot(a_i)))) * (std::abs(a_i.dot(f)) / std::sqrt(f.dot(f)));
    return lhs <= rhs;
}

Eigen::VectorXd truncatedGradient(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, double alpha_lb, double alpha_ub, double alpha_f) {
    //The code assumes as invariant that the dimension of f and A match and that y = Poisson(|Af|^2), i.e. the standard phase retrieval setup.
    int n = A.cols();
    int m = A.rows();

    Eigen::VectorXd result = Eigen::VectorXd::Zero(A.cols());
    for(size_t ix = 0; ix < A.rows(); ix++) {
        if(inE1(f, A.row(ix), alpha_lb, alpha_ub, n) && inE2(f, y, A, alpha_f, n, m, ix)) {
            double scalar = (std::pow(std::abs(A.row(ix).dot(f)), 2) - y[ix]) / f.dot(A.row(ix));
            result += scalar * A.row(ix);
        }
    }
    return result;
}

Eigen::VectorXd truncGradientDescent(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, double mu, int maxIter, double eps, double alpha_lb, double alpha_ub, double alpha_f) {
    int m = A.rows();
    for(size_t ix = 0; ix < maxIter; ix++) {
        Eigen::VectorXd grad = truncatedGradient(f, y, A, alpha_lb, alpha_ub, alpha_f);
        f = f - (mu / m) * grad;
        if(std::sqrt(grad.dot(grad)) < eps) {
            break;
        }
    }
    return f;
}

PYBIND11_MODULE(truncatedwf, m) {
    m.def("truncatedGradient", &truncatedGradient);
    m.def("truncGradientDescent", &truncGradientDescent);
}