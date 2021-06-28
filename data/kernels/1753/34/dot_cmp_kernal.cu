#include "includes.h"
__global__ void dot_cmp_kernal(const float* data1, const float* data2, float* device_soln, const int size, const int num_threads, const int offset)
{
float dot = 0.0f;

int idx = threadIdx.x + blockIdx.x*num_threads + offset;

for(int i = 0; i < size; i++){
int index = i*size + idx % size + ((idx/size)*size*size); //for coalesing
dot += data1[index]*data2[index];
}

device_soln[idx] = dot/size;

}