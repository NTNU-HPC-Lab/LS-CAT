#ifndef _CRBM_KERNEL_H
#define _CRBM_KERNEL_H

#include <curand_kernel.h>
#include <cuda.h>

#define MAX_FILETER_SIZE 8
#define MAX_POOLING_RATE 3
#define MAX_IMGAG_SIZE 128
#define RAND_SIZE 10000

__global__ void convolution_forward_kernel(float *input, 
        float *filters, float *feature_map, float *hbias, int input_size, 
        int channel_num, int feature_map_size, int filter_size,
        int filter_num, int lu_padding, float sigma);

__global__ void max_pooling_kernel(float *feature_map, float *probs, float *target,
        int feature_map_size, int feature_map_num, int pooling_rate,
        float *rnd_array, int rnd_num);

__global__ void convolution_backward_kernel(float *y_h, float *filters, float *vbias,
        float *target, float *y_v,
        int input_size, int lu_padding, int channel_num, int feature_map_size, 
        int filter_num, int filter_size, float *rnd_array, int rnd_num);

__global__ void compute_d_w_kernel(float *v, float *h, float *dw, bool is_init, 
        int input_size, int lu_padding, int channel_num, int filter_num, 
        int filter_size, int feature_map_size);

#endif
