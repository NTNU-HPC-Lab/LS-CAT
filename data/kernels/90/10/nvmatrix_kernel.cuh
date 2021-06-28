#ifndef _NVMATRIX_KERNEL_H
#define _NVMATRIX_KERNEL_H
#include <cuda.h>

#define NUM_THREAD_PER_ROW  128

__global__ void _init_mat(float *m, float val, int len);
__global__ void _copy_mat(float *m, float* target, int len);
__global__ void _ele_scale(float *m, float *target, float scaler, int len);
__global__ void _ele_add(float *m, float *target, float val, int len);
__global__ void _mat_add(float *ma, float *mb, float *target, float sa, float sb, int len);
__global__ void _mat_mul(float *ma, float *mb, float *target, int len);
__global__ void _mat_sum_col(float *m, float *target,int nrow, int ncol);
__global__ void _mat_sum_row(float *m, float *target,int nrow, int ncol);
__global__ void _mat_sum_row_fast(float *m, float *target, int nrow, int ncol, int agg_col);

#endif
