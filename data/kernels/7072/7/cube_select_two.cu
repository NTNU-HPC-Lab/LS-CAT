#include "includes.h"
__global__ void cube_select_two(int b, int n,float radius, const float* xyz, int* idx_out) {
int batch_idx = blockIdx.x;
xyz += batch_idx * n * 3;
idx_out += batch_idx * n * 16;
float temp_dist[16];
float judge_dist = radius * radius;
for(int i = threadIdx.x; i < n;i += blockDim.x) {
float x = xyz[i * 3];
float y = xyz[i * 3 + 1];
float z = xyz[i * 3 + 2];
for(int j = 0;j < 16;j ++) {
temp_dist[j] = judge_dist;
idx_out[i * 16 + j] = i; // if not found, just return itself..
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
int temp_idx = _x * 8 + _y * 4 + _z * 2;
bool flag = false;
for(int k = 0;k < 2;k ++) {
if (dist < temp_dist[temp_idx + k]) {
flag = true;
}
if (flag) {
for (int kk = 1; kk >= k + 1; kk --) {
idx_out[i * 16 + temp_idx + kk] = idx_out[i * 16 + temp_idx + kk - 1];
temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
}
idx_out[i * 16 + temp_idx + k] = j;
temp_dist[temp_idx + k] = dist;
break;
}
}

}
}
}