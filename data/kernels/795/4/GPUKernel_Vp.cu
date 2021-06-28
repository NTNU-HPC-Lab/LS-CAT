#include "includes.h"
__device__ int  GPUKernel_Position(int i,int j) {
if (i<j){
return j*(j+1)/2+i;
}
return i*(i+1)/2+j;
}
__global__ void GPUKernel_Vp(int a, int v,double * in,double * out) {

int blockid      = blockIdx.x*gridDim.y + blockIdx.y;
int id      = blockid*blockDim.x + threadIdx.x;

if ( id >= v*v*v ) return;

int  d = id%v;
int  b = (id-d)%(v*v)/v;
int  c = (id-d-b*v)/(v*v);

if ( b < a ) return;
if ( d > c ) return;

int cd   = GPUKernel_Position(c,d);
int vtri = v*(v+1)/2;

out[(b-a)*vtri+cd] = in[(b-a)*v*v+d*v+c] + in[(b-a)*v*v+c*v+d];
}