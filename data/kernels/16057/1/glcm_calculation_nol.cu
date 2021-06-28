#include "includes.h"
__global__ void glcm_calculation_nol(int *A,int *glcm, const int nx, const int ny,int maxx)
{

int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * nx + ix;
//unsigned int idr = iy * (maxx+1) + ix;
int k,l;
int p;
//Calculate GLCM
if(idx < nx*ny ){
for(k=0;k<=maxx;k++){
for(l=0;l<=maxx;l++){
if((A[idx]==k) && (A[idx+1]==l)){
p=((maxx+1)*k) +l;
atomicAdd(&glcm[p],1);
}
}
}
}

}