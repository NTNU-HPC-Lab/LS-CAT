#include "includes.h"
__global__ void pictureKernel(float* d_pix,int X, int Y) {
int thread_x=blockDim.x*blockIdx.x+threadIdx.x;
int thread_y=blockDim.y*blockIdx.y+threadIdx.y;
//	printf("thread_x=%d,blockDim.x=%d,blockIdx.x=%d,threadIdx=%d\n",thread_x,blockDim.x,blockIdx.x,threadIdx.x);
//	printf("thread_y=%d,blockDim.y=%d,blockIdx.y=%d,threadIdy=%d\n",thread_y,blockDim.y,blockIdx.y,threadIdx.y);
//	use this printf nvcc -arch compute_20 pixel.cu
if(thread_x<X&&thread_y<Y)	{
d_pix[thread_y*X+thread_x]*=2;
}
}