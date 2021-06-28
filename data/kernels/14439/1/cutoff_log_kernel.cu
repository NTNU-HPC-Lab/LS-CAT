#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void cutoff_log_kernel(double* device_array, double min_signal){
int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
if (device_array[thread_id] < min_signal){
device_array[thread_id] = logf(min_signal);
}
else{
device_array[thread_id] = logf(device_array[thread_id]);
}
}