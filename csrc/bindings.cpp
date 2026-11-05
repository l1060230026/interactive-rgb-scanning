#include "gn.h"
std::vector<std::vector<double>> take_linear_solver_probe();
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("gauss_newton_calib",&gauss_newton_calib_cuda);
  m.def("take_solver_probe",&take_linear_solver_probe);
}
