#include "includes.h"
__device__ float compute(int idx, float* buf, int s)
{
// some random calcs to make the kernel unempty
float k=0.0f;
for (int x=0;x<s;x++ ){
k+=cosf(x*0.1f*idx);
buf[x]=k;
}
for (int x=0;x<s/2;x++){
buf[x]=buf[x]*buf[x];
}
float sum=0.0f;
for (int x=s-1;x>=1;x--) {
sum += buf[x-1]/(fabsf(buf[x])+0.1f);
}
return sum;
}
__global__ void testWithGlobal(int n, int s, float* result, float* buf) {
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < n) {
result [idx] = compute(idx, &buf [idx * s],s);
}
}