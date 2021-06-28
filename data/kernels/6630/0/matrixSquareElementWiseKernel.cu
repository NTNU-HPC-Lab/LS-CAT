#include "includes.h"




using namespace std;

__device__ int getGlobalIdx_2D_2D()
{
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = blockId * (blockDim.x * blockDim.y)
+ (threadIdx.y * blockDim.x)
+ threadIdx.x;
return threadId;
}
__global__ void matrixSquareElementWiseKernel(float* in, float* out, int n, int m){
extern __shared__ float Rs[];

int index = getGlobalIdx_2D_2D();
if (index < n*m){

out[index] = in[index] * in[index];

}
}