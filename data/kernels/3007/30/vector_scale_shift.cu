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
__global__ void vector_scale_shift (const int n, const REAL* x, const int offset_x, const int stride_x, const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb, REAL* y, const int offset_y, const int stride_y) {

const int gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid < n) {
y[offset_y + gid * stride_y] = scalea * x[offset_x + gid * stride_x] + shifta;

}
}