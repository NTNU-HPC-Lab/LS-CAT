#include "includes.h"








__global__ void MatAdd(float *A, float *B, float *C, int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;

if((ix<nx)&&(iy<ny)){
C[idx]=A[idx]+B[idx];
}

}