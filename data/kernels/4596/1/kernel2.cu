#include "includes.h"
// Copyright 2019, Dimitra S. Kaitalidou, All rights reserved


#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32



__global__ void kernel2(int* D, int* Q){

int i = blockIdx.x * blockDim.x + threadIdx.x;

// Assign the values of the output array back to the input array
D[i] = Q[i];
}