#include "includes.h"
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
__global__ void accel_update(int nx, int ny, double dx2inv, double dy2inv, double* d_z, double* d_a) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
double ax, ay;
int r = i / nx;
int c = i % nx;
if(i < nx*ny) {
if(r<ny-1 && r>0 && c<nx-1 && c>0){
ax = (d_z[i+nx]+d_z[i-nx]-2.0*d_z[i])*dx2inv;
ay = (d_z[i+1]+d_z[i-1]-2.0*d_z[i])*dy2inv;
d_a[i] = (ax+ay)/2;
}
else
d_a[i] = 0.0;
}
}