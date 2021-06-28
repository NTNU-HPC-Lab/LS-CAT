#include "includes.h"

#define NUM_THREADS 512


__global__ void opt_cond_itr(int num_train_cases, double *opt_cond, double alpha_high, double alpha_high_prev, int high_label, int high_indx, double alpha_low, double alpha_low_prev, int low_label, int low_indx, double *kernel_val_mat){

int global_id = blockIdx.x * blockDim.x + threadIdx.x;

if(global_id < num_train_cases){
opt_cond[global_id] += (alpha_high - alpha_high_prev) * high_label * kernel_val_mat[high_indx*num_train_cases+global_id]
+ (alpha_low - alpha_low_prev) * low_label * kernel_val_mat[low_indx*num_train_cases+global_id];
}
}