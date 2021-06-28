#include "includes.h"
__global__ void correctBounds(double *d_ub, double *d_lb, int nRxns, double *d_prevPoint, double alpha, double beta, double *d_centerPoint, double *points, int pointsPerFile, int pointCount, int index){
int newindex = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for(int i=newindex;i<nRxns ;i+=stride){
if(points[pointCount+pointsPerFile*i]>d_ub[i]){
points[pointCount+pointsPerFile*i]=d_ub[i];
}else if(points[pointCount+pointsPerFile*i]<d_lb[i]){
points[pointCount+pointsPerFile*i]=d_lb[i];
}
d_prevPoint[nRxns*index+i]=points[pointCount+pointsPerFile*i];
d_centerPoint[nRxns*index+i]=alpha*d_centerPoint[nRxns*index+i]+beta*points[pointCount+pointsPerFile*i];
}
}