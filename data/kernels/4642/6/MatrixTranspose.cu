#include "includes.h"
__global__ void MatrixTranspose(float *a,float *b,int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;
int odx= iy*nx + ix;

if((ix<nx)&&(iy<ny)){
b[odx]=a[idx];
}

}