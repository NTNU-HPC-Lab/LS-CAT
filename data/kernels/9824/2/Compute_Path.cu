#include "includes.h"
__global__ void Compute_Path(int *Md, const int Width, const int k)
{
//2 Thread ID
int ROW = blockIdx.x;
int COL = threadIdx.x;


if (Md[ROW * Width + COL] > Md[ROW * Width + k] + Md[k * Width + COL])
Md[ROW * Width + COL] = Md[ROW * Width + k] + Md[k * Width + COL];
}