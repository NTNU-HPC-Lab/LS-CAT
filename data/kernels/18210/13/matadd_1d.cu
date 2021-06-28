#include "includes.h"
__global__ void matadd_1d(const float *a, const float *b, float *c, int n, int m){
int i = blockDim.x * blockIdx.x + threadIdx.x;
//处理m个数据相加
if(i < n){
for(int j = 0; j < m; j++){
int idx = j * n + i;
c[idx] = a[idx] + b[idx];
}
}
}