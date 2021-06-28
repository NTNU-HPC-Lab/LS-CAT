#include "includes.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif


#define n 10
#define W 100

cudaError_t knapsackCuda(int *output, const int *val, const int *wt, unsigned int size);

__device__ int maxi(int a, int b) {
return (a > b)? a : b;
}
__global__ void knapsackKernel(int *wt, int *val, int *output, int i) {
int w = threadIdx.x;

//__syncthreads();
if (i == 0 || w == 0)
output[(i*W)+w] = 0;
else if (wt[i-1] <= w)
output[(i*W)+w] = maxi(val[i-1] + output[((i-1)*W)+(w-wt[i-1])],  output[((i-1)*W)+w]);
else
output[(i*W)+w] = output[((i-1)*W)+w];
__syncthreads();

}