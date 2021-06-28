#include "includes.h"

using namespace std;

#define TILE_WIDTH 2



// main fn
__global__ void MatrixMult(int m, int n, int k, float *a, float *b, float *c)
{

int row = threadIdx.y + blockIdx.y*blockDim.y;
int col = threadIdx.x + blockIdx.x*blockDim.x;

if((row < m) && (col < k))
{
float temp = 0.0;
for (int i = 0; i < n; ++i)
{
temp += a[row*n+i]*b[col+i*k];
}
c[row*k+col] = temp;
}

}