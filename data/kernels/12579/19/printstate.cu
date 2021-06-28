#include "includes.h"
__global__ void printstate(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, int tnow){
const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
if(serial < n){
printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %d\n", serial, x[serial], y[serial], z[serial], vx[serial], vy[serial], vz[serial], tnow);
}
}