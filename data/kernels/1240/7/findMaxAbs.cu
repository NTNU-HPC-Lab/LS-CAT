#include "includes.h"
__global__ void findMaxAbs(int nRxns, double *d_umat2, int nMets, int *d_rowVec, int *d_colVec, double *d_val, int nnz, double *points, int pointsPerFile, int pointCount, int index){
int newindex = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for(int k=newindex;k<nnz;k+=stride){
d_umat2[nMets*index+d_rowVec[k]]+=d_val[k]*points[pointCount+pointsPerFile*d_colVec[k]];
}

}