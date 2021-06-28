#include "includes.h"
__global__ void Match2(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
__shared__ float buffer1[M2W*NDIM];  //%%%%
__shared__ float buffer2[M2H*NDIM];  //%%%%
__shared__ float scores[M2W*M2H];    //%%%%
int tx = threadIdx.x;
int ty = threadIdx.y;
int idx = tx + M2W*ty;
int bp1 = M2W*blockIdx.x;
if (ty<M2W)
for (int d=tx;d<NDIM;d+=M2W)
for (int j=ty;j<M2W;j+=M2H)
buffer1[j*NDIM + d] = d_pts1[(bp1 + j)*NDIM + d];   //%%%%
__syncthreads();

float max_score = 0.0f;
int index = -1;
for (int bp2=0;bp2<NPTS;bp2+=M2H) {
for (int d=tx;d<NDIM;d+=M2W)
buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d]; //%%%%
__syncthreads();

float score = 0.0f;
for (int d=0;d<NDIM;d++)
score += buffer1[tx*NDIM + d]*buffer2[ty*NDIM + d];   //%%%%
scores[idx] = score;
__syncthreads();

if (ty==0) {
for (int i=0;i<M2H;i++) {
if (scores[i*M2W + tx]>max_score) {
max_score = scores[i*M2W + tx];
index = bp2 + i;
}
}
}
__syncthreads();
}

if (ty==0) {
d_score[bp1 + tx] = max_score;
d_index[bp1 + tx] = index;
}
}