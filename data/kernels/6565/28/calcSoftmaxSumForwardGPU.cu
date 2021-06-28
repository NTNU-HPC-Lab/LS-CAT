#include "includes.h"
__global__ void calcSoftmaxSumForwardGPU(float *array, float *out, float *max, float *sum, int *mutex, int batch_size, int in_size_x, unsigned n)
{
unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int stride = gridDim.x * blockDim.x;
unsigned int offset = 0;

// __shared__ float cache[ 32 ][ BLOCK ]; // max 0xc000
extern __shared__ float cache[];

float temp = 0.0;
while(index + offset < n){
// float v = exp(array[index + offset] - *max);
float v = exp(array[index + offset] - *(max + blockIdx.x));
out[index + offset] = v;
temp = temp + v;
offset += stride;
}

// cache[threadIdx.x] = temp;
cache[index] = temp;

__syncthreads();

unsigned int prev_i = blockDim.x;
unsigned int i = blockDim.x / 2;

while ( i!=0 ){
if(threadIdx.x < i){
cache[index] = cache[index] + cache[index + i];
}
if(prev_i%2 != 0){
cache[blockIdx.x * blockDim.x + 0] = cache[blockIdx.x * blockDim.x + 0] + cache[blockIdx.x * blockDim.x + prev_i-1];
}
__syncthreads();
prev_i = i;
i /= 2;
}

if( threadIdx.x == 0 ){
while( atomicCAS(mutex, 0, 1) != 0 );
// *sum = *sum + cache[blockIdx.x][0];
*(sum+blockIdx.x) = *(sum+blockIdx.x) + cache[blockIdx.x * blockDim.x + 0];
atomicExch(mutex, 0);
}

/* original
float sum = 0.0;
for ( int i = 0; i < in.size.x; ++i ){
float v = in( b, i, 0, 0 );
v = exp(v - max_v);
out( b, i, 0, 0 ) = v;
sum += v;
}
*/
}