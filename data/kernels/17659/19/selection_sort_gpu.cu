#include "includes.h"
__global__ void selection_sort_gpu(int b, int n, int m, int k, float *dist, int *idx, float *val) {
int batch_index = blockIdx.x;
dist+=m*n*batch_index;
idx+=m*k*batch_index;
val+=m*k*batch_index;

int index = threadIdx.x;
int stride = blockDim.x;

float *p_dist;
for (int j=index;j<m;j+=stride) {
p_dist = dist+j*n;
// selection sort for the first k elements
for (int s=0;s<k;++s) {
int min=s;
// find the min
for (int t=s+1;t<n;++t) {
if (p_dist[t]<p_dist[min]) {
min = t;
}
}
// update idx and val
idx[j*n+s] = min;
val[j*n+s] = p_dist[min];
// swap min-th and i-th element
float tmp = p_dist[min];
p_dist[min] = p_dist[s];
p_dist[s] = tmp;
}
}
}