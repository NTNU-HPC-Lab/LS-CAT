#include "includes.h"
__global__ void OPT_3_HIST(int* lcm, int* hist, int* lcmsizes, int n) {

//
int vertex = blockIdx.x;
int vcomp = threadIdx.x;
bool equal;

//
__shared__ int cval;

//
if(vcomp == 0)
cval = 0;
__syncthreads();

//
if(vertex < n && vcomp < n)
for(int i = vcomp; i < n; i += blockDim.x) {

if(vertex == i) {
atomicAdd(&cval, 1);
continue;
}

if((lcmsizes[vertex+1] - lcmsizes[vertex]) != (lcmsizes[i+1] - lcmsizes[i]))
continue;

equal = false;

for(int j = 0; j < lcmsizes[vertex+1] - lcmsizes[vertex]; j++) {

if(lcm[lcmsizes[vertex] + j] == lcm[lcmsizes[i] + j])
equal = true;

else {
equal = false;
break;
}
}

if(equal)
atomicAdd(&cval, 1);
}

__syncthreads();
if(vertex < n && vcomp == 0 && cval > 0) {
atomicAdd(&hist[cval], 1);
//printf("\nv%d: %d\n", vertex, cval);
}
}