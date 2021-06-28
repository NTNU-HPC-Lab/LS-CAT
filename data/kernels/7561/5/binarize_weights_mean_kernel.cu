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


__global__ void binarize_weights_mean_kernel(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int f = i / size;
if (f >= n) return;
float mean = mean_arr_gpu[f];
binary[i] = (weights[i] > 0) ? mean : -mean;
}