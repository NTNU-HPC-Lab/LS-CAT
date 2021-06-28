#include "includes.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")
#endif

extern "C" {
}

extern "C" {
double get_time_point();
void start_timer();
void stop_timer();
double get_time();
void stop_timer_and_show();
void stop_timer_and_show_name(char *name);
void show_total_time();
}


__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (s >= size) return;
int i = 0;
float mean = 0;
for(i = 0; i < n; ++i){
mean += fabs(input[i*size + s]);
}
mean = mean / n;
for(i = 0; i < n; ++i){
binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
}
}