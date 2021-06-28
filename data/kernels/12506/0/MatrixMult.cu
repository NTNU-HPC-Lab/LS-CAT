#include "includes.h"

#define TILE_WIDTH 40

//-----------------------------------------------



//--------------------------------------------------

// Compute C = A * B

//-------------------------------------------------

__global__ void MatrixMult(int m, int n, int k, double *a, double *b, double *c)
{

int row = threadIdx.y + blockIdx.y*blockDim.y;
int col = threadIdx.x + blockIdx.x*blockDim.x;

if((row < m) && (col < k))
{
double temp = 0.0;
for (int i = 0; i < n; ++i)
{
temp += a[row*n+i]*b[col+i*k];
}
c[row*k+col] = temp;
}

}