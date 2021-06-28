#include "includes.h"
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
__global__ void pos_update(int nx, int ny, double dt, double* d_z, double* d_v, double* d_a) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
int r = i / nx;
int c = i % nx;
if(r<ny-1 && r>0 && c<nx-1 && c>0){
d_v[i] = d_v[i] + dt*d_a[i];
d_z[i] = d_z[i] + dt*d_v[i];
}
}