#include "includes.h"
__global__ void TestHomographies(float *d_coord, float *d_homo, int *d_counts, int numPts, float thresh2)
{
__shared__ float homo[8*TESTHOMO_LOOPS];
__shared__ int cnts[TESTHOMO_TESTS*TESTHOMO_LOOPS];
const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int idx = blockIdx.y*blockDim.y + tx;
const int numLoops = blockDim.y*gridDim.y;
if (ty<8 && tx<TESTHOMO_LOOPS)
homo[tx*8+ty] = d_homo[idx+ty*numLoops];
__syncthreads();
float a[8];
for (int i=0;i<8;i++)
a[i] = homo[ty*8+i];
int cnt = 0;
for (int i=tx;i<numPts;i+=TESTHOMO_TESTS) {
float x1 = d_coord[i+0*numPts];
float y1 = d_coord[i+1*numPts];
float x2 = d_coord[i+2*numPts];
float y2 = d_coord[i+3*numPts];
float nomx = __fmul_rz(a[0],x1) + __fmul_rz(a[1],y1) + a[2];
float nomy = __fmul_rz(a[3],x1) + __fmul_rz(a[4],y1) + a[5];
float deno = __fmul_rz(a[6],x1) + __fmul_rz(a[7],y1) + 1.0f;
float errx = __fmul_rz(x2,deno) - nomx;
float erry = __fmul_rz(y2,deno) - nomy;
float err2 = __fmul_rz(errx,errx) + __fmul_rz(erry,erry);
if (err2<__fmul_rz(thresh2,__fmul_rz(deno,deno)))
cnt ++;
}
int kty = TESTHOMO_TESTS*ty;
cnts[kty + tx] = cnt;
__syncthreads();
int len = TESTHOMO_TESTS/2;
while (len>0) {
if (tx<len)
cnts[kty + tx] += cnts[kty + tx + len];
len /= 2;
__syncthreads();
}
if (tx<TESTHOMO_LOOPS && ty==0)
d_counts[idx] = cnts[TESTHOMO_TESTS*tx];
__syncthreads();
}