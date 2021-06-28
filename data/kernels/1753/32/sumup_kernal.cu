#include "includes.h"
__global__ void sumup_kernal(const float * data, float * device_stats, const int size, const int dim2size, const int num_threads, const int offset)
{
float sum = 0.0f;

const uint x=threadIdx.x;
const uint y=blockIdx.x;

int idx = x + y*num_threads + offset;

for(int i = 0; i < size; i++){
int index = i*dim2size + idx % dim2size;
sum += data[index];
}

device_stats[idx] = sum/size;
}