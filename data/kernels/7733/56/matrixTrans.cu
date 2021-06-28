#include "includes.h"
__global__ void matrixTrans(float * M,float * MT)
{
int val=0;

int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

MT[row + col*N] = 0;
if (row < N && col < N)
{
val = M[col + row*N];
MT[row + col*N] = val;

}
}