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


__global__ void set_zero_kernel(float *src, int size)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) src[i] = 0;
}