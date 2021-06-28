#include "includes.h"
__global__ void dot_cmp_kernaldm(const float* data1, const float* data2, const float* dm, float* device_soln, const int size, const int num_threads, const int offset)
{
float dot = 0.0f;
float nnn = 0.0f;

int idx = threadIdx.x + blockIdx.x*num_threads + offset;

for(int i = 0; i < size; i++){
int index = i*size + idx % size + ((idx/size)*size*size); //for coalesing
if(dm[index] > 0.5){
dot += data1[index]*data2[index];
nnn += 1.0f;
}
}

device_soln[idx] = dot/nnn;

}