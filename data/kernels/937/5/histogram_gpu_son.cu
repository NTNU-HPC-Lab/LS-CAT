#include "includes.h"
__global__ void histogram_gpu_son(unsigned char * d_img, unsigned int * d_hist,  int img_size,  int serialNum)
{
// __shared__ unsigned int aa[ROLLSIZE][256];
extern __shared__ unsigned int aa[];
int x = threadIdx.x + blockDim.x*blockIdx.x;
int i;

for(i = 0; i < ROLLSIZE; i++) aa[(i << 8) + threadIdx.x] = 0;
__syncthreads();

int end = (x+1)*serialNum;
if (end >= img_size) end = img_size;

for(i = x*serialNum; i < end; i++) atomicAdd(&(aa[((threadIdx.x >> 4 ) << 8) +  d_img[i]]), 1);
__syncthreads();

unsigned int s;
for(s = 16 / 2; s > 0; s >>= 1) {
//Only when numThreads == 256
for(i = 0; i < s; i++) aa[(i << 8) + threadIdx.x] += aa[((i+s) << 8) + threadIdx.x];


// if (threadIdx.x < s) {
// for(i = 0; i < 256; i++) {
//     aa[threadIdx.x][i] += aa[threadIdx.x + s][i];
// }
// }
__syncthreads();
}

atomicAdd(&(d_hist[threadIdx.x]),aa[threadIdx.x]);
return;
}