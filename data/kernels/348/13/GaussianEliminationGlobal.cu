#include "includes.h"
__global__ void GaussianEliminationGlobal(const int clusterSize,float *x, const float *diagonal_values , const float *non_diagonal_values ,float *y , const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x ;
const int gi = index * clusterSize;

float matrix[180][180];	//size of matrix
for (int i = gi; i < gi + clusterSize;++i)
{
for (int j = gi; j < gi + clusterSize;++j)
{
matrix[i][j]=0;
}
matrix[i][i] = diagonal_values[i];
}
for(int i = gi; i < gi + clusterSize - 1 ;++i)
{
matrix[i][i+1] = non_diagonal_values[2*i+1];
matrix[i+1][i] = non_diagonal_values[2*i+2];
}

// triangle form
for (int i = gi ; i < gi + clusterSize; ++i)
{
//for every row...
for (int j = i+1; j < gi + clusterSize; ++j)
{
//calculate ratio for every row below it using the triangular
double ratio = matrix[j][i] / matrix[i][i];
for(int k = gi; k < gi + clusterSize; ++k)
{
//Eliminate every column based on that ratio
matrix[j][k] = matrix[j][k] - (matrix[i][k] * ratio);
}
//elimination on the coefficient vector
y[j] = y[j] - (y[i] * ratio);
}
}
__syncthreads();
//Back substitution
for (int i = gi + clusterSize-1; i > gi-1; --i)
{
double current = 0;
for (unsigned int j = i; j < gi + clusterSize; ++j)
{
current = current + (matrix[i][j] * x[j]);
}
x[i] = (y[i] - current) / matrix[i][i];
}
}