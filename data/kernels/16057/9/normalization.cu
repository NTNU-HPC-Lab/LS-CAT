#include "includes.h"
__global__ void normalization(int *glcm,float *norm,int max,int sum){
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
__syncthreads();
if(idx<(max+1)*(max+1)){
norm[idx]=float(glcm[idx])/float(sum);
}
}