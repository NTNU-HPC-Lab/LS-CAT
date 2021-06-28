/*
 * Copyright (c) 2016-present Jean-Noel Braun.
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

#ifndef BCNN_UTILS_H
#define BCNN_UTILS_H

#include "bcnn/bcnn.h"

/* OpenMP */
#ifdef BCNN_USE_OPENMP
#include <omp.h>
#endif
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

#include <stdio.h>
#include <stdlib.h>  // RAND_MAX
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int state;
    float r;
} bcnn_gauss_gen;

typedef struct {
    bcnn_log_callback fct;
    bcnn_log_level lvl;
} bcnn_log_context;

/** Convenient macros */
#define BCNN_CHECK(exp, err) \
    do {                     \
        if (!(exp)) {        \
            return (err);    \
        }                    \
    } while (0)

#define BCNN_CHECK_AND_LOG(ctx, exp, err, fmt, ...)                \
    do {                                                           \
        if (!(exp)) {                                              \
            bcnn_log((ctx), BCNN_LOG_ERROR, (fmt), ##__VA_ARGS__); \
            return (err);                                          \
        }                                                          \
    } while (0)

#define BCNN_CHECK_STATUS(s)         \
    do {                             \
        bcnn_status ret = (s);       \
        if ((ret) != BCNN_SUCCESS) { \
            return (ret);            \
        }                            \
    } while (0)

#define BCNN_ERROR(ctx, err, fmt, ...)                         \
    do {                                                       \
        bcnn_log((ctx), BCNN_LOG_ERROR, (fmt), ##__VA_ARGS__); \
        return (err);                                          \
    } while (0)

#define BCNN_INFO(ctx, fmt, ...)                              \
    do {                                                      \
        bcnn_log((ctx), BCNN_LOG_INFO, (fmt), ##__VA_ARGS__); \
    } while (0)

#define BCNN_WARNING(ctx, fmt, ...)                              \
    do {                                                         \
        bcnn_log((ctx), BCNN_LOG_WARNING, (fmt), ##__VA_ARGS__); \
    } while (0)

#define BCNN_PARSE_CLEANUP(l, t, n)     \
    do {                                \
        for (int i = 0; i < (n); ++i) { \
            bh_free((t[i]));            \
        }                               \
        bh_free(t);                     \
        bh_free(l);                     \
    } while (0)

float bcnn_rng_gaussian(bcnn_gauss_gen *g);

void bcnn_log(bcnn_log_context ctx, bcnn_log_level level, const char *fmt, ...);

void bcnn_draw_color_box(unsigned char *img, int w_img, int h_img, float cx_box,
                         float cy_box, float w_box, float h_box,
                         unsigned char color[3]);

static inline int bcnn_rand_between(int min, int max) {
    if (min > max) {
        return 0.f;
    }
    return (int)(((float)rand() / RAND_MAX * (max - min)) + min + 0.5f);
}

static inline int bcnn_omp_get_num_threads() {
#ifdef BCNN_USE_OPENMP
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
#else
    return 1;
#endif
}

static inline const char *bcnn_act2str(bcnn_activation a) {
    switch (a) {
        case BCNN_ACT_TANH:
            return "Tanh";
        case BCNN_ACT_RELU:
            return "ReLU";
        case BCNN_ACT_RAMP:
            return "Ramp";
        case BCNN_ACT_SOFTPLUS:
            return "Softplus";
        case BCNN_ACT_LRELU:
            return "Leaky-ReLU";
        case BCNN_ACT_ABS:
            return "AbsVal";
        case BCNN_ACT_CLAMP:
            return "Clamp";
        case BCNN_ACT_PRELU:
            return "PReLU";
        default:
            return "None";
    }
}

#ifdef BCNN_USE_CUDA

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
#define BCNN_CUDA_THREADS 1024
#else
#define BCNN_CUDA_THREADS 512
#endif

static inline int bcnn_cuda_blocks(const int n) {
    return (n - 1) / (BCNN_CUDA_THREADS) + 1;
}

cublasHandle_t bcnn_cublas_handle();

#define bcnn_cuda_check(RET)                                                   \
    {                                                                          \
        if ((RET) != cudaSuccess) {                                            \
            fprintf(stderr, "[ERROR] [CUDA] %s\n", cudaGetErrorString((RET))); \
            exit((RET));                                                       \
        }                                                                      \
    }
#define bcnn_cublas_check(RET)                                    \
    {                                                             \
        if ((RET) != CUBLAS_STATUS_SUCCESS) {                     \
            fprintf(stderr, "[ERROR] [CUBLAS] %d\n", (int)(RET)); \
            exit((RET));                                          \
        }                                                         \
    }

#define bcnn_curand_check(RET)                                    \
    {                                                             \
        if ((RET) != CURAND_STATUS_SUCCESS) {                     \
            fprintf(stderr, "[ERROR] [CURAND] %d\n", (int)(RET)); \
            exit((RET));                                          \
        }                                                         \
    }

int *bcnn_cuda_malloc_i32(int n);

float *bcnn_cuda_malloc_f32(int n);

float *bcnn_cuda_memcpy_f32(float *x, int n);

void bcnn_cuda_memcpy_f32_noalloc(float *x, float *x_gpu, int n);

void bcnn_cuda_fill_with_random(float *x_gpu, int n);

void bcnn_cuda_free(void *x_gpu);

void bcnn_cuda_memcpy_host2dev(float *x_gpu, float *x, int n);

void bcnn_cuda_memcpy_dev2host(float *x_gpu, float *x, int n);

void bcnn_cuda_set_device(int id);

#ifdef BCNN_USE_CUDNN
#define bcnn_cudnn_check(RET)                       \
    {                                               \
        if ((RET) != CUDNN_STATUS_SUCCESS) {        \
            fprintf(stderr, "[ERROR] [CUDNN] %s\n", \
                    cudnnGetErrorString((RET)));    \
            exit((RET));                            \
        }                                           \
    }
cudnnHandle_t bcnn_cudnn_handle();
#endif  // BCNN_USE_CUDNN

#endif  // BCNN_USE_CUDA

#ifdef __cplusplus
}
#endif

#endif  // BCNN_UTILS_H