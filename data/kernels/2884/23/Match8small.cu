#include "includes.h"
__global__ void Match8small(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
__shared__ float4 buffer1[M7W*NDIM/4];
__shared__ float4 buffer2[M7H*NDIM/4];
int tx = threadIdx.x;
int ty = threadIdx.y;
int bp1 = M7W*blockIdx.x;
for (int d=tx;d<NDIM/4;d+=M7W)
for (int j=ty;j<M7W;j+=M7H/M7R/NRX)
buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

float max_score[NRX];
int index[NRX];
for (int i=0;i<NRX;i++) {
max_score[i] = 0.0f;
index[i] = -1;
}
int idx = ty*M7W + tx;
int ix = idx%(M7W/NRX);
int iy = idx/(M7W/NRX);
for (int bp2=0;bp2<NPTS;bp2+=M7H) {
for (int d=tx;d<NDIM/4;d+=M7W)
for (int j=ty;j<M7H;j+=M7H/M7R/NRX)
buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
__syncthreads();

float score[M7R][NRX];
for (int dy=0;dy<M7R;dy++)
for (int i=0;i<NRX;i++)
score[dy][i] = 0.0f;
for (int d=0;d<NDIM/4;d++) {
float4 v1[NRX];
for (int i=0;i<NRX;i++)
v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
for (int dy=0;dy<M7R;dy++) {
float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];
for (int i=0;i<NRX;i++) {
score[dy][i] += v1[i].x*v2.x;
score[dy][i] += v1[i].y*v2.y;
score[dy][i] += v1[i].z*v2.z;
score[dy][i] += v1[i].w*v2.w;
}
}
}
for (int dy=0;dy<M7R;dy++) {
for (int i=0;i<NRX;i++) {
if (score[dy][i]>max_score[i]) {
max_score[i] = score[dy][i];
index[i] = bp2 + M7R*iy + dy;
}
}
}
__syncthreads();
}

float *scores = (float*)buffer1;
int *indices = (int*)&scores[M7W*M7H/M7R];
if (idx<M7W*M7H/M7R/NRX) {
for (int i=0;i<NRX;i++) {
scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];
indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
}
}
__syncthreads();

if (ty==0) {
float max_score = scores[tx];
int index = indices[tx];
for (int y=0;y<M7H/M7R;y++)
if (scores[y*M7W + tx]>max_score) {
max_score = scores[y*M7W + tx];
index = indices[y*M7W + tx];
}
d_score[bp1 + tx] = max_score;
d_index[bp1 + tx] = index;
}
}