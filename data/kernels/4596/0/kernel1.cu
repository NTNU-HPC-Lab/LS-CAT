#include "includes.h"
// Copyright 2019, Dimitra S. Kaitalidou, All rights reserved


#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32



__global__ void kernel1(int* D, int* Q, int k){

// Find index
int i = blockIdx.x * blockDim.x + threadIdx.x;
int block = (int)(i / (2 * k));
int j;

if(i % 2 == 0) j = 2 * block * k + (int)(i / 2) - k * ((int)(i / (2 * k)));
else j = (2 * block + 1) * k + (int)(i / 2) - k * ((int)(i / (2 * k)));

// Assign the values to the output array
Q[j] = D[i];
}