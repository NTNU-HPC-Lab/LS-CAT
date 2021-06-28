#include "includes.h"

#define MAX_THREADS 20
#define pi(x) printf("%d\n",x);
#define HANDLE_ERROR(err) ( HandleError( err, __FILE__, __LINE__ ) )
#define th_p_block  256


__global__ void dotPro(long n, float *vec1, float *vec2, float *vec3) {

__shared__ float cache[th_p_block];
unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int cacheIdx =  threadIdx.x;
float temp = 0;
while(tid < n)
{
temp += vec1[tid] * vec2[tid];
tid += blockDim.x * gridDim.x;
}

cache[cacheIdx] = temp;
__syncthreads();

// reduction
unsigned i = blockDim.x/2; // need the num threads to be a power of two (256 is okay)
while( i != 0 ){
if(cacheIdx < i){
cache[cacheIdx] += cache[cacheIdx + i ];
}

__syncthreads(); //sync threads in the current block
// power of two needed here
i = i/2;
}
if(cacheIdx == 0){
vec3[blockIdx.x] = cache[0];
}
//    if (tid < n) vec3[i] = vec1[i] * vec2[i];
}