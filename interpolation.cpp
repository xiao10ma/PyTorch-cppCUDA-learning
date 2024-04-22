#include "utils.h"

torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor point
){
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_fw_cu(feats, point);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_interpolation", &trilinear_interpolation);
}