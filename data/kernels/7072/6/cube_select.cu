#include "includes.h"
__global__ void cube_select(int b, int n,float radius, const float* xyz, int* idx_out) {
int batch_idx = blockIdx.x;
xyz += batch_idx * n * 3;
idx_out += batch_idx * n * 8;
float temp_dist[8];
float judge_dist = radius * radius;
for(int i = threadIdx.x; i < n;i += blockDim.x) {
float x = xyz[i * 3];
float y = xyz[i * 3 + 1];
float z = xyz[i * 3 + 2];
for(int j = 0;j < 8;j ++) {
temp_dist[j] = 1e8;
idx_out[i * 8 + j] = i; // if not found, just return itself..
}
for(int j = 0;j < n;j ++) {
if(i == j) continue;
float tx = xyz[j * 3];
float ty = xyz[j * 3 + 1];
float tz = xyz[j * 3 + 2];
float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
if(dist > judge_dist) continue;
int _x = (tx > x);
int _y = (ty > y);
int _z = (tz > z);
int temp_idx = _x * 4 + _y * 2 + _z;
if(dist < temp_dist[temp_idx]) {
idx_out[i * 8 + temp_idx] = j;
temp_dist[temp_idx] = dist;
}
}
}
}