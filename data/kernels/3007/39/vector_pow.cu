#include "includes.h"
extern "C" {

#ifndef REAL
#define REAL float
#endif

#ifndef CAST
#define CAST(fun) fun ## f
#endif

#ifndef REAL2o3
#define REAL2o3 (REAL)0.6666666666666667
#endif

#ifndef REAL3o2
#define REAL3o2 (REAL)1.5
#endif

























































































































































































}
__global__ void vector_pow (const int n, const REAL* x, const int offset_x, const int stride_x, const REAL* y, const int offset_y, const int stride_y, REAL* z, const int offset_z, const int stride_z) {

const int gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid < n) {
z[offset_z + gid * stride_z] =
CAST(pow)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
}
}