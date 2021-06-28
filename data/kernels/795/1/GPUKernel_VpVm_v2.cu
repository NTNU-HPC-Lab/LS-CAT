#include "includes.h"
__device__ int  GPUKernel_Position(int i,int j) {
if (i<j){
return j*(j+1)/2+i;
}
return i*(i+1)/2+j;
}
__global__ void GPUKernel_VpVm_v2(int a, int b,int v,double * in,double * outp,double * outm) {

int blockid = blockIdx.x*gridDim.y + blockIdx.y;
int id      = blockid*blockDim.x + threadIdx.x;

int v2 = v*v;

if ( id >= v2 ) return;

int  d = id%v;
int  c = (id-d)/v;

if ( d > c ) return;

int cd   = GPUKernel_Position(c,d);

outp[cd] = in[d*v+c] + in[c*v+d];
outm[cd] = in[d*v+c] - in[c*v+d];
}