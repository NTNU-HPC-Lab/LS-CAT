#include "includes.h"
__global__ void GPUKernel_Iqdb(int a,int v,int nQ,double * in,double * out) {

int blockid = blockIdx.x*gridDim.y + blockIdx.y;
int id      = blockid*blockDim.x + threadIdx.x;

if ( id >= v*v*nQ ) return;

int  q = id%nQ;
int  d = (id-q)%(nQ*v)/nQ;
int  b = (id-q-d*nQ)/(nQ*v);

if ( b < a ) return;

int id2 = (b-a)*nQ*v+d*nQ+q;
out[id2] = in[id];

}