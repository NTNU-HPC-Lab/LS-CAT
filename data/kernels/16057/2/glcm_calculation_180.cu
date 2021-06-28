#include "includes.h"
__global__ void glcm_calculation_180(int *A,int *glcm, const int nx, const int ny,int max){
//int iy = threadIdx.y + blockIdx.y* blockDim.y;
unsigned int idx =blockIdx.x*nx+threadIdx.x;
int i;
int k=0;
for(i=0;i<nx;i++){
if(idx>=i*nx && idx<((i+1) *nx)-1){
k=max*A[idx+1]+A[idx];
atomicAdd(&glcm[k],1);
}
}
}