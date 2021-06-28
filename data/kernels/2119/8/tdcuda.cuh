#include <stdio.h>
#include <iostream>
#include <random>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

#define VEC_SIZE 1024*1024*16
#define DIV_MAX 1024*16
#define DIV_MIN 1
#define POW_MAX 15
#define MAX_ITER 1

int run_exercices();
void exercice2_cuda_withmemcpy(float* d_x, float* x, float* d_y, float* y, float* d_s, float* s, int k, int size);
void exercice2_cuda_nomemcpy(float* d_x, float* d_y, float* d_s, int k, int size);
void exercice2_cpu(float a, float* x, float* y, float* s, int size);

__global__ void saxpy_kernel(int n, float a, float *v1, float *v2, float *s);
__global__ void mean_kernel(int n, float* v1, float* v2, float* res);

float rand_float();