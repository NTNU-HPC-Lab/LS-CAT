#include "includes.h"
__global__ void stats_kernal(const float *data, float * device_soln, const int size, const int num_calcs, const int num_threads, const int offset)
{

float sum = 0.0f;
float sum_sq = 0.0f;

int idx = threadIdx.x + blockIdx.x*num_threads + offset;

for(int i = 0; i < size; i++){
int index = i*size + idx % size + ((idx/size)*size*size); //for coalesing
float datum = data[index]; //so we dno't need multiple accesses to global mem, I would think that the compiler would optimize this, but the manual said to program like this....
sum += datum;
sum_sq += datum*datum;
}

device_soln[idx] = sum;
device_soln[idx + num_calcs] = sum_sq;

}