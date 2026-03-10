#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;

py::array_t<double> truncatedGradient(py::array_t<double> f, py::array_t<double> y, Eigen::MatrixXd A, int n, double alpha_lb, double alpha_ub, double alpha_f) {
    //The code assumes as invariant that the dimension of f and A match and that y = Poisson(|Af|^2), i.e. the standard phase retrieval setup.
    return [0.0]
}
//A.row(i) gives the row
PYBIND11_MODULE(truncatedwf, m) {
    m.def("truncatedGradient", &truncatedGradient);
}