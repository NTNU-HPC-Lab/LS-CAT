#include "includes.h"
__global__ void Match7(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
__shared__ float4 buffer1[M7W*NDIM/4]; //%%%%
__shared__ float4 buffer2[M7H*NDIM/4];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bp1 = M7W*blockIdx.x;
for (int d=tx;d<NDIM/4;d+=M7W)
for (int j=ty;j<M7W;j+=M7H/M7R)      //%%%%
buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

float max_score = 0.0f;
int index = -1;
for (int bp2=0;bp2<NPTS;bp2+=M7H) {
for (int d=tx;d<NDIM/4;d+=M7W)
for (int j=ty;j<M7H;j+=M7H/M7R)       //%%%%
buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
__syncthreads();

float score[M7R];
for (int dy=0;dy<M7R;dy++)
score[dy] = 0.0f;
for (int d=0;d<NDIM/4;d++) {
float4 v1 = buffer1[tx*NDIM/4 + (d + tx)%(NDIM/4)];
for (int dy=0;dy<M7R;dy++) {
float4 v2 = buffer2[(M7R*ty + dy)*(NDIM/4) + d];
score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
}
}
for (int dy=0;dy<M7R;dy++) {
if (score[dy]>max_score) {
max_score = score[dy];
index = bp2 + M7R*ty + dy;
}
}
__syncthreads();
}

float *scores = (float*)buffer1;
int *indices = (int*)&scores[M7W*M7H/M7R];
scores[ty*M7W + tx] = max_score;
indices[ty*M7W + tx] = index;
__syncthreads();

if (ty==0) {
max_score = scores[tx];
index = indices[tx];
for (int y=0;y<M7H/M7R;y++)
if (scores[y*M7W + tx]>max_score) {
max_score = scores[y*M7W + tx];
index = indices[y*M7W + tx];
}
d_score[bp1 + tx] = max_score;
d_index[bp1 + tx] = index;
}
}