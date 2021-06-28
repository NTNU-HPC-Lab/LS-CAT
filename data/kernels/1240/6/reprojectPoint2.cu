#include "includes.h"
__global__ void reprojectPoint2(double *d_N, int nRxns, int istart, double *d_umat, double *points, int pointsPerFile, int pointCount,int index){
int newindex= blockIdx.x * blockDim.x + threadIdx.x;
int stride= blockDim.x * gridDim.x;

for(int i=newindex;i<nRxns;i+=stride){
points[pointCount+pointsPerFile*i]=0;
for(int j=0;j<nRxns-istart;j++){
points[pointCount+pointsPerFile*i]+=d_N[j*nRxns+i]*d_umat[nRxns*index+j];//here N*tmp
}
}
}