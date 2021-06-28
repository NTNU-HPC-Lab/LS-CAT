#include "includes.h"
__global__ void pw_scatter_cu_z(double *c, const double *pwcc, const double scale, const int ngpts, const int nmaps, const int *ghatmap) {

const int igpt =
(gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

if (igpt < ngpts) {
c[2 * ghatmap[igpt]] = scale * pwcc[2 * igpt];
c[2 * ghatmap[igpt] + 1] = scale * pwcc[2 * igpt + 1];
if (nmaps == 2) {
c[2 * ghatmap[igpt + ngpts]] = scale * pwcc[2 * igpt];
c[2 * ghatmap[igpt + ngpts] + 1] = -scale * pwcc[2 * igpt + 1];
}
}
}