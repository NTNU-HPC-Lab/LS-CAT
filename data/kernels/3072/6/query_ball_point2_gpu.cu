#include "includes.h"
__global__ void query_ball_point2_gpu(int b, int n, int m, int nsample, const float *xyz1, const float *xyz2, const float *radii, int *idx, int *pts_cnt) {
int batch_index = blockIdx.x;
xyz1 += n*3*batch_index;
xyz2 += m*3*batch_index;
radii += m*batch_index;
idx += m*nsample*batch_index;  // m clusters, each having nsamples
pts_cnt += m*batch_index; // counting how many unique points selected in local region

int index = threadIdx.x;
int stride = blockDim.x;

for (int j=index;j<m;j+=stride) {  // index of cluster
int cnt = 0;
for (int k=0;k<n;++k) {  // index of point
if (cnt == nsample)
break; // only pick the FIRST nsample points in the ball
float x2=xyz2[j*3+0];
float y2=xyz2[j*3+1];
float z2=xyz2[j*3+2];
float x1=xyz1[k*3+0];
float y1=xyz1[k*3+1];
float z1=xyz1[k*3+2];
float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
if (d<radii[j]) {
if (cnt==0) { // set ALL indices to -1, s.t. we know which points are padded
for (int l=0;l<nsample;++l)
idx[j*nsample+l] = k;
}
idx[j*nsample+cnt] = k;
cnt+=1;
}
}
pts_cnt[j] = cnt;
}
}