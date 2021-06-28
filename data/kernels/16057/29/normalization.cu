#include "includes.h"
__global__ void normalization(int *glcm,float *norm,int Max,int sum){
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * Max + ix;
__syncthreads();
if(idx<(Max+1)*(Max+1)){
norm[idx]=float(glcm[idx])/float(sum);
}
}