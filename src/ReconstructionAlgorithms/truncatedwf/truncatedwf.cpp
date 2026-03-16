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
/*
Eigen::VectorXd truncSpectralInit(Eigen::VectorXd y, RowMatrixXd A, double alpha_f) {
    int n = A.cols();
    int m = A.rows();
    double alpha_f_sq = std::pow(alpha_f, 2);

    double lamba_0_sq = y.sum() / m;
    Eigen::VectorXd condY = y.unaryexpr(std::ptr_fun(truncInitFilter));

    RowMatrixXd Y = RowMatrixXd::Zero(n ,n);

    return y; //Placeholder.
}
*/

std::tuple<int, int> truncCountingSpectralInit(Eigen::VectorXd y, double alpha_f, int n, int m) {
    double lambda_0_sq = y.sum() / m;
    double alpha_f_sq = std::pow(alpha_f, 2);

    int nonZeroCount = 0;
    int truncatedCount = 0; 

    for(size_t ix = 0; ix < y.size(); ix++) {
        if(y[ix] != 0) {
            nonZeroCount += 1;
            if(std::abs(y[ix]) > alpha_f_sq * lambda_0_sq) {
                truncatedCount += 1;
            }
        }
    }
    return std::make_tuple(truncatedCount, nonZeroCount);
}

bool inE1(double conditionRatio, double alpha_lb, double alpha_ub, int n) {
    return (alpha_lb <= conditionRatio) && (conditionRatio <= alpha_ub);
}

double calcKt(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, int m) {
    //This assumes real phase retrieval, i.e. input vectors contain real values.
    double sum = 0;
    for(int l = 0; l < m; l++) {
        sum += std::abs(y[l] - std::pow(std::abs(A.row(l).dot(f)), 2));
    }
    return sum / m;
}

bool inE2(Eigen::VectorXd f, Eigen::VectorXd y, Eigen::VectorXd a_i, double alpha_f, double conditionRatio, int i, double Kt) {
    double lhs = std::abs(y[i] - std::pow(a_i.dot(f), 2));

    double rhs = alpha_f * Kt * conditionRatio;
    return lhs <= rhs;
}

Eigen::VectorXd truncatedGradient(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, double alpha_lb, double alpha_ub, double alpha_f) {
    //The code assumes as invariant that the dimension of f and A match and that y = Poisson(|Af|^2), i.e. the standard phase retrieval setup.
    int n = A.cols();
    int m = A.rows();

    double Kt = calcKt(f, y, A, m);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(A.cols());
    for(size_t ix = 0; ix < A.rows(); ix++) {
        Eigen::VectorXd a_i = A.row(ix);
        double conditionRatio = (std::sqrt(static_cast<float>(n) / a_i.dot(a_i))) * (std::abs(a_i.dot(f)) / f.norm());
        if(inE1(conditionRatio, alpha_lb, alpha_ub, n) && inE2(f, y, a_i, alpha_f, conditionRatio, ix, Kt)) {
            double ai_dot_f = a_i.dot(f);
            double scalar = (ai_dot_f * ai_dot_f - y[ix]) / ai_dot_f;
            result += scalar * A.row(ix);
        }
    }
    return result;
}

Eigen::VectorXd truncGradientDescent(Eigen::VectorXd f, Eigen::VectorXd y, RowMatrixXd A, double mu, int maxIter, double eps, double alpha_lb, double alpha_ub, double alpha_f) {
    int m = A.rows();
    double stepSize = mu/m;

    for(size_t ix = 0; ix < maxIter; ix++) {
        Eigen::VectorXd grad = truncatedGradient(f, y, A, alpha_lb, alpha_ub, alpha_f);
        f = f - stepSize * grad;
        if(grad.dot(grad) < eps) {
            break;
        }
    }
    return f;
}

PYBIND11_MODULE(truncatedwf, m) {
    m.def("truncatedGradient", &truncatedGradient);
    m.def("truncGradientDescent", &truncGradientDescent);
    m.def("truncCountingSpectralInit", &truncCountingSpectralInit);
}