#include "includes.h"
__global__ void pw_gather_cu_z(      double *pwcc, const double *c, const double  scale, const int     ngpts, const int    *ghatmap) {

const int igpt = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

if (igpt < ngpts) {
pwcc[2 * igpt    ] = scale * c[2 * ghatmap[igpt]    ];
pwcc[2 * igpt + 1] = scale * c[2 * ghatmap[igpt] + 1];
}
}