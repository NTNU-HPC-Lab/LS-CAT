#include "includes.h"
__global__ void leapstep(unsigned long n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt) {
const unsigned long serial = blockIdx.x * blockDim.x + threadIdx.x;
if (serial < n){
x[serial] += dt * vx[serial];
y[serial] += dt * vy[serial];
z[serial] += dt * vz[serial];
}
}