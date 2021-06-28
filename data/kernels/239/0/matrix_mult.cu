#include "includes.h"

// CUDA runtime


// Helper functions and utilities to work with CUDA


#define N 256
//#define M 256


//__global__ÉùÃ÷µÄº¯Êý£¬¸æËß±àÒëÆ÷Õâ¶Î´úÂë½»ÓÉCPUµ÷ÓÃ£¬ÓÉGPUÖ´ÐÐ

__global__ void matrix_mult(float *dev_a, float* dev_b, float* dev_c, int Width)
{
int Row = blockIdx.y*blockDim.y+threadIdx.y;
int Col = blockIdx.x*blockDim.x+threadIdx.x;
if ((Row < Width) && (Col < Width)) {
float Pvalue = 0;
for (int k = 0; k < Width; k++)
{
Pvalue += dev_a[Row*Width + k] * dev_b[k*Width+Col];
}
dev_c[Row*Width + Col] = Pvalue;

}
}