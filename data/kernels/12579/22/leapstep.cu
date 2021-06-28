#include "includes.h"
__global__ void leapstep(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
const unsigned int serial = blockIdx.x * BLOCKSIZE + threadIdx.x;
if(serial < n){
x[serial] += dt * vx[serial];
y[serial] += dt * vy[serial];
z[serial] += dt * vz[serial];
}
}