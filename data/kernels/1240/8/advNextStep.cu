#include "includes.h"
__global__ void advNextStep(double *d_prevPoint, double *d_umat, double d_stepDist, int nRxns, double *points, int pointsPerFile, int pointCount, int index){
int newindex= blockIdx.x * blockDim.x + threadIdx.x;
int stride= blockDim.x * gridDim.x;

for(int i=newindex;i<nRxns;i+=stride){
points[pointCount+pointsPerFile*i]=d_prevPoint[nRxns*index+i]+d_stepDist*d_umat[nRxns*index+i];
}
}