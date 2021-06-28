#include "includes.h"
__global__ void Match8blocked(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
#define NUM (NRX*M7R)                       // 32*8 threads
__shared__ float4 buffer1[M7W*NDIM/4];    // 32*32
__shared__ float4 buffer2[M7H*NUM];       // 32*8
int tx = threadIdx.x;
int ty = threadIdx.y;
int bp1 = M7W*blockIdx.x;
for (int d=tx;d<NDIM/4;d+=M7W)
for (int j=ty;j<M7W;j+=M7H/M7R)
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
float score[M7R][NRX];
for (int dy=0;dy<M7R;dy++)
for (int i=0;i<NRX;i++)
score[dy][i] = 0.0f;

int d = (idx%NUM);
int j = (idx/NUM);
buffer2[j*NUM + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
__syncthreads();
for (int dp=0;dp<NDIM/4;dp+=NUM) {
float4 temp;
if (dp<(NDIM/4-NUM))
temp = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + dp + d + NUM];

if (idx<M7W*M7H/M7R/NRX) {
for (int d=0;d<NUM;d++) {
float4 v1[NRX];
#pragma unroll
for (int i=0;i<NRX;i++)
v1[i] = buffer1[(((M7W/NRX)*i + ix)<<5) + ((dp + d + (M7W/NRX)*i + ix)&31)];
//v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (dp + d + (M7W/NRX)*i + ix)%(NDIM/4)];
#pragma unroll
for (int dy=0;dy<M7R;dy++) {
float4 v2 = buffer2[(M7R*iy + dy)*NUM + d];
#pragma unroll
for (int i=0;i<NRX;i++) {
score[dy][i] += v1[i].x*v2.x;
score[dy][i] += v1[i].y*v2.y;
score[dy][i] += v1[i].z*v2.z;
score[dy][i] += v1[i].w*v2.w;
}
}
}
}
__syncthreads();

if (dp<(NDIM/4-NUM)) {
buffer2[j*NUM + d] = temp;
__syncthreads();
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