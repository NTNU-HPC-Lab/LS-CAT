#include "includes.h"
/*
* This project is dual licensed. You may license this software under one of the
following licences:

+ Creative Commons Attribution-Share Alike 3.0 Unported License
http://creativecommons.org/licenses/by-nc-sa/3.0/

+ GNU GENERAL PUBLIC LICENSE v3, designated as a "BY-SA Compatible License"
as defined in BY-SA 4.0 on 8 October 2015

* See the LICENSE file in the root directory of this source tree for full
copyright disclosure, and other details.
*/


/* Header files */




/* Constants */

#define threads 256 /* It's the number of threads we are going to use per block on the GPU */

using namespace std;


/* Kernels */

/* This kernel counts the number of pairs in the data file */
/* We will use this kernel to calculate real-real pairs and random-random pairs */


/* This kernel counts the number of pairs that there are between two data groups */
/* We will use this kernel to calculate real-random pairs and real_1-real_2 pairs (cross-correlation) */
/* NOTE that this kernel has NOT been merged with 'binning' above: this is for speed optimization, we avoid passing extra variables to the GPU */


__global__ void binning(float *xd,float *yd,float *zd,float *ZZ,int number_lines,int points_per_degree, int number_of_degrees)
{

/* We define variables (arrays) in shared memory */

float angle;
__shared__ float temp[threads];

/* We define an index to run through these two arrays */

int index = threadIdx.x;

/* This variable is necesary to accelerate the calculation, it's due that "temp" was definied in the shared memory too */

temp[index]=0;
float x,y,z; //MCM
float xx,yy,zz; //MCM

/* We start the counting */

for (int i=0;i<number_lines;i++)
{
x = xd[i];//MCM
y = yd[i];//MCM
z = zd[i];//MCM

/* The "while" replaces the second for-loop in the sequential calculation case (CPU). We use "while" rather than "if" as recommended in the book "Cuda by Example" */

for(int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
dim_idx < number_lines;
dim_idx += blockDim.x * gridDim.x)
{
xx = xd[dim_idx];//MCM
yy = yd[dim_idx];//MCM
zz = zd[dim_idx];//MCM

/* We make the dot product */
angle = x * xx + y * yy + z * zz;//MCM


//angle[index]=xd[i]*xd[dim_idx]+yd[i]*yd[dim_idx]+zd[i]*zd[dim_idx];//MCM
//__syncthreads();//MCM

/* Sometimes "angle" is higher than one, due to numnerical precision, to solve it we use the next sentence */

angle=fminf(angle,1.0);
angle=acosf(angle)*180.0/M_PI;
//__syncthreads();//MCM

/* We finally count the number of pairs separated an angular distance "angle", always in shared memory */

if(angle < number_of_degrees)
{
atomicAdd( &temp[int(angle*points_per_degree)], 1.0);
}
__syncthreads();
}
}

/* We copy the number of pairs from shared memory to global memory */

atomicAdd( &ZZ[threadIdx.x] , temp[threadIdx.x]);
__syncthreads();
}