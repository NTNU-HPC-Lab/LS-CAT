#include "includes.h"
__global__ void pw_copy_cr_cu_z(const double *zin, double *dout, const int n) {
const int igpt =
(gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

if (igpt < n) {
dout[igpt] = zin[2 * igpt];
}
}