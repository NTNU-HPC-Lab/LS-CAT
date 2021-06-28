#include "includes.h"
__global__ void generateCoefficients(float *chromosomes, const int chromSize, const float* noise, const int population, const int alpha){

int i;

// For up to a 1D grid of 3D blocks...
int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

curandState st;
curand_init((int)noise[threadGlobalID] << threadGlobalID, threadGlobalID * (threadGlobalID == population - 1 ? noise[0] : noise[threadGlobalID]), 0, &st);

if (threadGlobalID > 0){
for (i = 0; i < chromSize; i++){
if (curand_uniform(&st) < 0.5){
chromosomes[chromSize*threadGlobalID + i] = curand_uniform(&st) *alpha;
}
else{
chromosomes[chromSize*threadGlobalID + i] = -1 * curand_uniform(&st) * alpha;
}
}
}
}