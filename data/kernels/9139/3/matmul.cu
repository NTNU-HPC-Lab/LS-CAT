#include "includes.h"
__global__ void matmul(double *a, double *b, double *c, int n)
{
// Get global thread ID
int Col = blockIdx.x*blockDim.x+threadIdx.x;
int Row = blockIdx.y*blockDim.y+threadIdx.y;
// Not out of bounds
if((Col<n) && (Row<n)) {// Mutliply matrices
// printf("Hello thread %d\n", threadIdx.x);
// c[Row*n + Col] = 0;
double sum = 0.0;
for(int k=0;k<n;k++) {
// c[Row*n + Col] += a[Row*n+k]*b[k*n+Col];
sum += a[Row*n+k]*b[k*n+Col];
}
c[Row*n + Col] = sum;
}
}