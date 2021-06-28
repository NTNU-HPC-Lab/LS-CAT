#include "includes.h"
const int listLength = 700;
__global__ void squareKernel(float* d_in, float *d_out, int threads_num) {
const unsigned int lid = threadIdx.x; // local id inside a block
const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
if (gid < threads_num){
d_out[gid] = powf((d_in[gid]/(d_in[gid]-2.3)),3);
}// do computation
}