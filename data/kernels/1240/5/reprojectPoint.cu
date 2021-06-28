#include "includes.h"
__global__ void reprojectPoint(double *d_N, int nRxns, int istart, double *d_umat, double *points, int pointsPerFile, int pointCount, int index){
int newindex = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for(int i=newindex;i<nRxns-istart;i+=stride){
d_umat[nRxns*index+i]=0;//d_umat now is d_tmp
for(int j=0;j<nRxns;j++){
d_umat[nRxns*index+i]+=d_N[j+i*nRxns]*points[pointCount+pointsPerFile*j];//here t(N)*Pt
}
}
}