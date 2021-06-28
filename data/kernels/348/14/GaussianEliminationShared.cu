#include "includes.h"
__global__ void GaussianEliminationShared(const int clusterSize,float *x, const float *diagonal_values , const float *non_diagonal_values ,float *y )
{
const int index = blockIdx.x ;

__shared__ float shared_m[9][9]; // size of cluster
for (int i = 0; i < clusterSize;++i)
{
for (int j = 0; j < clusterSize;++j)
{
shared_m[i][j]=0;
}
}
for(int i = 0; i < clusterSize; ++i)
{
shared_m[i][i] = diagonal_values[clusterSize * index + i];
}
for(int i = 0; i < clusterSize-1;++i)
{
shared_m[i][i+1] = non_diagonal_values[clusterSize * index * 2 + 2*i+1];
shared_m[i+1][i] = non_diagonal_values[clusterSize * index * 2 + 2*i+2];
}

// triangle form
for (int i = 0 ; i < clusterSize; ++i)
{
//for every row...
for (int j = i+1; j < clusterSize; ++j)
{
//calculate ratio for every row below it using the triangular
double ratio = shared_m[j][i] / shared_m[i][i];
for(int k = 0; k < clusterSize; ++k)
{
//Eliminate every column based on that ratio
shared_m[j][k] = shared_m[j][k] - (shared_m[i][k] * ratio);
}
//elimination on the coefficient vector
y[clusterSize * index +j] = y[clusterSize * index +j] - (y[clusterSize * index +i] * ratio);
}
}
__syncthreads();

//Back substitution
for (int i = clusterSize-1; i > -1; --i)
{
double current = 0;
for (unsigned int j = i; j < clusterSize; ++j)
{
current = current + (shared_m[i][j] * x[clusterSize * index +j]);
}
x[clusterSize * index +i] = (y[clusterSize * index +i] - current) / shared_m[i][i];
}

}