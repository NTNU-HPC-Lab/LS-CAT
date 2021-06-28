#include "includes.h"
#ifdef INFINITY
/* INFINITY is supported */
#endif

float **A, **D, *d2; //Table A distance, D minimum distance,d2 tempTable 1-d

__global__ void calc(float *d_D, int n, int k){
int i = blockIdx.x * blockDim.x + threadIdx.x;   //We find i & j in the Grid of threads
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (d_D[i + j*n] > d_D[i + k*n] + d_D[k + j*n]) d_D[i + j*n] = d_D[i + k*n] + d_D[k + j*n];  //Every thread calculates its proper value
}