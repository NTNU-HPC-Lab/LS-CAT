#include "includes.h"
__global__ void query_ball_point_gpu(int b, int n, int m, const float* radius, int nsample, const float *xyz1, const float *xyz2, int *idx) {
for (int i=0;i<b;++i) {
for (int j=0;j<m;++j) {
int cnt = 0;
for (int k=0;k<n;++k) {
if (cnt == nsample)
break; // only pick the FIRST nsample points in the ball
float x2=xyz2[j*3+0];
float y2=xyz2[j*3+1];
float z2=xyz2[j*3+2];
float x1=xyz1[k*3+0];
float y1=xyz1[k*3+1];
float z1=xyz1[k*3+2];
float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
if (d<radius[0]) {
if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
for (int l=0;l<nsample;++l)
idx[j*nsample+l] = k;
}
idx[j*nsample+cnt] = k;
cnt+=1;
}
}
}
xyz1+=n*3;
xyz2+=m*3;
idx+=m*nsample;
}
}