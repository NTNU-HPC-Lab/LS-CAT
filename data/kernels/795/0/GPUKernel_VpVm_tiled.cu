#include "includes.h"
__global__ void GPUKernel_VpVm_tiled(int a, int bstart, int bsize,int v,double * in,double * outp,double * outm) {

int blockid = blockIdx.x*gridDim.y + blockIdx.y;
int id      = blockid*blockDim.x + threadIdx.x;

int v2 = v*v;

if ( id >= v2*bsize ) return;

// id : b*v2+c*v+d

int  d = id%v;
int  c = (id-d)%(v*v)/v;

if ( d > c ) return;

//int  b = (id-d)%(v*bsize)/v;


//int  c = (id-d-b*v)/(bsize*v);
int  b = (id-d-c*v)/(v*v);

if ( b + bstart < a ) return;

int cd   = c*(c+1)/2 + d;
int vtri = v*(v+1)/2;
int bv2  = b*v2;

//outp[b*vtri+cd] = in[bv2+d*v+c] + in[bv2+c*v+d];
//outm[b*vtri+cd] = in[bv2+d*v+c] - in[bv2+c*v+d];
outp[b*vtri+cd] = in[bv2+d*v+c] + in[id];
outm[b*vtri+cd] = in[bv2+d*v+c] - in[id];
}