#include "includes.h"
__global__ void pw_copy_rc_cu_z(const double *din, double *zout, const int n) {
const int igpt =
(gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

if (igpt < n) {
zout[2 * igpt] = din[igpt];
zout[2 * igpt + 1] = 0.0e0;
}
}