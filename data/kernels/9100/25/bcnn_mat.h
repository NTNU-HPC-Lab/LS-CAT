/*
* Copyright (c) 2016 Jean-Noel Braun.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/


#ifndef BCNN_MAT_H
#define BCNN_MAT_H

/* Cuda include */
#ifdef BCNN_USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef BCNN_USE_CUDNN
#include <cudnn.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Matrix computation routines */
int bcnn_fill_f32(int n, float a, float *x);
int bcnn_copy_f32(int n, float *x, float *y);
int bcnn_axpy(int n, float a, float *x, float *y);
int bcnn_scal(int n, float a, float *x);
int bcnn_add_scalar(int n, float a, float *x);
int bcnn_pow(int n, float *x, float a, float *y);
float bcnn_dot(int n, float *x, float *y);
int bcnn_vsum(int n, float *x, float *sum);
int bcnn_vadd(int n, float *a, float *b, float *y);
int bcnn_vsub(int n, float *a, float *b, float *y);
int bcnn_vdiv(int n, float *a, float *b, float *y);
int bcnn_vmul(int n, float *a, float *b, float *y);
int bcnn_axpby(int n, float a, float *x, float b, float *y);
int bcnn_gemv(int trans_a, int m, int n, float alpha, float *a, float *x,
    float beta, float *y);
int bcnn_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);
int bcnn_xnor_gemm(int trans_a, int trans_b, int M, int N, int K, float ALPHA,
                        unsigned int *A, int lda,
                        unsigned int *B, int ldb,
                        float BETA,
                        float *C, int ldc);
float bcnn_l2_distance(float *x, float *y, int n);
float bcnn_sqrdiff_vs(float *x, float a, int n);
float bcnn_shiftdot(int n, float *x, float a, float *y, float b);
int bcnn_varnorm(int n, float *a, float c, float *y);
int bcnn_varmean(int n, float *m, float a, float *var);
void bcnn_add_bias(float *output, float *bias, int batch_size, int num_channels, int spatial_size);
void bcnn_grad_bias(float *grad_bias, float *grad_data, int batch_size, int num_channels, int spatial_size);
void bcnn_im2col(const float *data_im, const int channels, const int height, const int width,
    const int kernel_size, const int pad, const int stride, float *data_col);
void bcnn_col2im(const float *data_col, const int channels, const int height, const int width,
    const int kernel, const int pad, const int stride, float *data_im);

/* Cuda kernels routines */
#ifdef BCNN_USE_CUDA

/* Math routines */
void bcnn_cuda_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha,
    float *a, int lda,
    float *b, int ldb,
    float beta,
    float *c, int ldc);
void bcnn_cuda_gemv(int trans_a, const int m,
    const int n, const float alpha, const float *a, const float *x,
    const float beta, float *y);
void bcnn_cuda_fill_f32(int n, float alpha, float *x, int incx);
void bcnn_cuda_copy_f32(int n, float * x, int incx, float * y, int incy);
void bcnn_cuda_axpy(int n, float alpha, float *x, int incx, float *y, int incy);
void bcnn_cuda_scal(int n, float alpha, float *x, int incx);
void bcnn_cuda_pow(int n, float *x, float a, float *y);
void bcnn_cuda_axpby(int n, float a, float *x, float b, float *y);
void bcnn_cuda_add_scalar(int n, float a, float* y);
void bcnn_cuda_vadd(int n, float *a, float *b, float *y);
void bcnn_cuda_vsub(int n, float *a, float *b, float *y);
void bcnn_cuda_vmul(int n, float *a, float *b, float *y);
void bcnn_cuda_vdiv(int n, float *a, float *b, float *y);

void bcnn_cuda_mean_variance_forward(float *x, int b, int c, int wxh, float *mean, float *var);
void bcnn_cuda_norm_forward(float *x, float *mean, float *variance, int b, int c, int wxh);
void bcnn_cuda_mean_variance_backward(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff);
void bcnn_cuda_norm_backward(float *x, float *mean, float *var, float *mean_diff, float *var_diff, int b, int c, int wxh, float *grad);

void bcnn_op_cuda_tanh(int n, float *x, float *y);
void bcnn_op_cuda_tanh_grad(int n, float *x, float *dx);
void bcnn_op_cuda_relu(int n, float *x, float *y);
void bcnn_op_cuda_relu_grad(int n, float *x, float *dx);
void bcnn_op_cuda_clamp(int n, float *x, float *y);
void bcnn_op_cuda_clamp_grad(int n, float *x, float *dx);
void bcnn_op_cuda_ramp(int n, float *x, float *y);
void bcnn_op_cuda_ramp_grad(int n, float *x, float *dx);

void bcnn_cuda_add_bias(float *output, float *bias, int batch_size, int num_channels, int spatial_size);
void bcnn_cuda_grad_bias(float *grad_bias, float *grad_data, int batch_size, int num_channels, int spatial_size);

void bcnn_cuda_im2col(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col);
void bcnn_cuda_col2im(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);

#endif


#ifdef __cplusplus
}
#endif

#endif // BCNN_MAT_H
