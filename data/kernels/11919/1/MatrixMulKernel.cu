#include "includes.h"
__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width) {
//2D Thread ID
int tx = threadIdx.x;
int ty = threadIdx.y;

//Pvalue stores the Pd element that is computed by the thread
float Pvalue = 0;

for(int k = 0; k < Width ; ++k) {
float Mdelement = Md[ty*Width + k];
float Ndelement = Nd[k*Width + tx];
Pvalue += (Mdelement*Ndelement);
}

Pd[ty*Width + tx] = Pvalue;
}