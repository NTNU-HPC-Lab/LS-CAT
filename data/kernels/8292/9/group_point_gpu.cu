#include "includes.h"
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
for (int i=0;i<b;++i) {
for (int j=0;j<m;++j) {
for (int k=0;k<nsample;++k) {
int ii = idx[j*nsample+k];
for (int l=0;l<c;++l) {
out[j*nsample*c+k*c+l] = points[ii*c+l];
}
}
}
points+=n*c;
idx+=m*nsample;
out+=m*nsample*c;
}
}