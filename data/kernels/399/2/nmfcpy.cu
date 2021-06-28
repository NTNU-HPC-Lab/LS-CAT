#include "includes.h"
using namespace std;

#define BLOCKSIZE 32

//test code
__global__ void nmfcpy(float *mat, float *matcp, int m, int n) //kernel copy must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

if (row < m && col < n)
mat[row*n+col] = matcp[row*n+col];
}