#include "includes.h"
__global__ void gpu_histo_kernel_naive(u_char* Source, int *res, unsigned height, unsigned width){
int j = blockIdx.x*blockDim.x + threadIdx.x;
int i = blockIdx.y*blockDim.y + threadIdx.y;
if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
else {
u_char val = Source[i*width+j];
atomicAdd(&res[val],1);
}
}