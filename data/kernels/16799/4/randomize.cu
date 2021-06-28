#include "includes.h"
__device__ unsigned char value(float n1, float n2, int hue) {
if (hue > 360) hue -= 360;
else if (hue < 0) hue += 360;

if (hue < 60)
return (unsigned char) (255 * (n1 + (n2 - n1) * hue / 60));
if (hue < 180)
return (unsigned char) (255 * n2);
if (hue < 240)
return (unsigned char) (255 * (n1 + (n2 - n1) * (240 - hue) / 60));
return (unsigned char) (255 * n1);
}
__global__ void randomize(float* array, curandState* rand, unsigned long N)
{
int x = threadIdx.x + (blockIdx.x * blockDim.x);
int y = threadIdx.y + (blockIdx.y * blockDim.y);
unsigned long tid = x + (y * blockDim.x * gridDim.x);

if(tid < N){
curandState localState = rand[tid]; // get local curandState as seed
float theRand = curand_uniform(&localState); // use to get value from 0-1
rand[tid] = localState; // save new state as previous state for next gen

array[tid] = theRand;
}

}