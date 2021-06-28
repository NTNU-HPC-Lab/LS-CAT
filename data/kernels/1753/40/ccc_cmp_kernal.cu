#include "includes.h"
__global__ void ccc_cmp_kernal(const float* data1, const float* data2, float* device_soln, const int size, const int num_calcs, const int num_threads, const int offset)
{
float avg1 = 0.0f;
float avg2 = 0.0f;
float var1 = 0.0f;
float var2 = 0.0f;
float ccc = 0.0f;

const uint x=threadIdx.x;
const uint y=blockIdx.x;

int idx = x + y*num_threads + offset;

for(int i = 0; i < size; i++){
int index = i*size + idx % size + ((idx/size)*size*size); //for coalesing
avg1 += data1[index];
avg2 += data2[index];
var1 += data1[index]*data1[index];
var2 += data2[index]*data2[index];
ccc += data1[index]*data2[index];
}

device_soln[idx] = avg1/size;
device_soln[idx + num_calcs] = avg2/size;
device_soln[idx + 2*num_calcs] = var1/size;
device_soln[idx + 3*num_calcs] = var2/size;
device_soln[idx + 4*num_calcs] = ccc/size;


}