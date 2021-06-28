#include "includes.h"
__global__ void glcm_calculation_90(int *A,int *glcm, const int nx, const int ny,int max){
int ix = threadIdx.x + blockIdx.x* blockDim.x;
int iy = threadIdx.y + blockIdx.y* blockDim.y;
unsigned int idx =iy*nx+ix;
int i;
int k=0;
for(i=0;i<nx-1;i++){
if(idx>=i*nx && idx<((i+1) *nx)){
k=max*A[idx+nx]+A[idx];
atomicAdd(&glcm[k],1);
}
}
__syncthreads();
}