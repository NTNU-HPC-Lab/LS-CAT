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


__global__ void binarize_kernel(float *x, int n, float *binary)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= n) return;
binary[i] = (x[i] >= 0) ? 1 : -1;
}