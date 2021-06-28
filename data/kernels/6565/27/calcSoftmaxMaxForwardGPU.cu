#include "includes.h"
__global__ void calcSoftmaxMaxForwardGPU(float *array, float *max, int *mutex, int batch_size, int in_size_x, unsigned n)
{
unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int stride = gridDim.x * blockDim.x; // = in_size_x
unsigned int offset = 0;

// __shared__ float cache[ 32 ][ BLOCK ]; // this should be constant. batch_size * in_size_x actually
extern __shared__ float cache[];
// printf("index=%d, stride=%d, n=%d, gridDim.x=%d, blockDim.x=%d\n", index, stride, n, gridDim.x, blockDim.x);

float temp = -1.0;
while(index + offset < n){
temp = fmaxf(temp, array[index + offset]);
offset += stride;
}

// cache[threadIdx.x] = temp;
cache[index] = temp;
__syncthreads();

unsigned int prev_i = blockDim.x;
unsigned int i = blockDim.x / 2;
while ( i!=0 ){
if(threadIdx.x < i){
// cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
cache[index] = fmaxf(cache[index], cache[index + i]);
}
if(prev_i%2 != 0){
cache[(blockIdx.x * blockDim.x)] = fmaxf(cache[(blockIdx.x * blockDim.x)], cache[(blockIdx.x * blockDim.x) + prev_i-1]);
}
__syncthreads();
i /= 2;
}

if( threadIdx.x == 0 ){
while( atomicCAS(mutex, 0, 1) != 0 ); // atomic compare and swap.
// *max = fmaxf(*max, cache[0]);
*(max+blockIdx.x) = fmaxf(*(max+blockIdx.x), cache[blockIdx.x * blockDim.x + 0]);
atomicExch(mutex, 0); // atomic exchange.
}

/* original
for ( int b = 0; b < in.size.b; ++b ){
float max_v = 0.0;
for ( int i = 0; i < in.size.x; ++i ){
float v = in( b, i, 0, 0 );
if(v>max_v){
max_v = v;
}
}
}
*/
}