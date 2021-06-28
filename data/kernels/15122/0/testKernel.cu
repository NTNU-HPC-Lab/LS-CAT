#include "includes.h"
/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/* matrix project which demonstrates the basics on how to setup a project
* example application.
* Device code.
*/

#ifndef _matrix_KERNEL_H_
#define _matrix_KERNEL_H_





#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

#endif // #ifndef _matrix_KERNEL_H_
__global__ void testKernel(	float* d_matrixA, float* d_matrixB, float* d_matrixC, const unsigned int ah, const unsigned int aw, const unsigned int bh, const unsigned int bw) {
// shared memory - Matrix B
#ifdef CHANGE4
__shared__ float shm_matrixB[KERNEL_SIZE+(2*KERNEL_LENGTH)];
#elif defined(CHANGE1)
__shared__ float shm_matrixB[KERNEL_SIZE];
#endif

// shared memory - SubMatrix A
#ifdef CHANGE4
__shared__ float shm_subMatrixA[BLOCK_SIZE_HEIGHT*BLOCK_SIZE_WIDTH+WARP_SIZE];

#elif defined(CHANGE3)
__shared__ float shm_subMatrixA0[BLOCK_SIZE_HEIGHT*BLOCK_SIZE_WIDTH];
__shared__ float shm_subMatrixA1[BLOCK_SIZE_HEIGHT*BLOCK_SIZE_WIDTH];

#elif defined(CHANGE2)
__shared__ float shm_subMatrixA[2*BLOCK_SIZE_HEIGHT*BLOCK_SIZE_WIDTH];

#endif

// the size is determined by the host application
const unsigned int bx = blockIdx.x;
const unsigned int by = blockIdx.y;


// access thread id
const int tx = threadIdx.x;
const int ty = threadIdx.y;

#ifdef CHANGE3
int xstep = bx;
int ystep = 2 * by;
#elif defined(CHANGE2)
int xstep = bx;
int ystep = by;
#else
int xstep = BLOCK_SIZE * bx;
int ystep = BLOCK_SIZE * by;
#endif

#ifdef CHANGE3
float sum0 = 0;
float sum1 = 0;

#else
float sum = 0;
#endif

int y = ystep + ty;
int x = xstep + tx;


#ifdef CHANGE4

if(tx<(KERNEL_LENGTH))
{// Padding zeros to get rid of dependence on divergence
shm_matrixB[ tx ] = 0;
shm_matrixB[ KERNEL_SIZE + tx ] = 0;
}

// Padding zeros to get rid of dependence on divergence
if(tx<(KERNEL_SIZE))
shm_matrixB[ tx + KERNEL_LENGTH ] = d_matrixB[ tx ];

if(tx<(WARP_SIZE))
shm_subMatrixA[ tx  ] = 0;

__syncthreads();


#elif defined(CHANGE1)

if((tx<(KERNEL_SIZE)))
shm_matrixB[ tx ] = d_matrixB[ tx ];
//	__syncthreads();


#endif


/* -------------------------------- Computation -------------------------------------*/

#ifdef CHANGE4
//modified code
for (int j=0; j<bh+1; j++) {


shm_subMatrixA[tx+WARP_SIZE] = 0;

if ((y-j+1)>-1)
{
shm_subMatrixA[tx+WARP_SIZE] = d_matrixA[(y-j+1)*aw+(x)];
}

__syncthreads();

for(int k = 0; k < bw; ++k) {
float b0 = shm_matrixB[j*bw+k];
float b1 = shm_matrixB[(j+1)*bw+k];
float a = 0;

a = shm_subMatrixA[tx-k+WARP_SIZE];
sum0 += a*b0;
sum1 += a*b1;
}//k loop
__syncthreads();
}//j loop


#elif defined(CHANGE3)
//modified code
for (int j=0; j<bh; j++) {

if ((((y-j)>-1) &&(y-j)<ah))
{
shm_subMatrixA0[tx] = d_matrixA[(y-j)*aw+(x)];
}
if ((((y+1-j)>-1) &&(y+1-j)<ah))
{
shm_subMatrixA1[tx] = d_matrixA[(y+1-j)*aw+(x)];
}

__syncthreads();

for(int k = 0; k < bw; ++k) {
float b = shm_matrixB[j*bw+k];
float a0 = 0;
float a1 = 0;
// check the out-of-bound
if ((((y-j)>-1) &&(y-j)<ah)&&(x-k)>-1&&(x-k)<aw) {

a0 = shm_subMatrixA0[tx-k];

sum0 += a0*b;
}
if ((((y+1-j)>-1) &&(y+1-j)<ah)&&(x-k)>-1&&(x-k)<aw) {

a1 = shm_subMatrixA1[tx-k];

sum1 += a1*b;
}
}//k loop
__syncthreads();
}//j loop


#elif defined(CHANGE2)
//modified code
for (int j=0; j<bh; j++) {

#if 0
if(tx<WARP_SIZE)
if (((y-j)>-1) &&((y-j)<ah)&&((x-DATA_TO_PULL_SIZE)>-1)&&((x - DATA_TO_PULL_SIZE)<aw))
shm_subMatrixA[tx] = d_matrixA[(y-j)*aw+(x-DATA_TO_PULL_SIZE)];
#endif

if ((((y-j)>-1) &&(y-j)<ah))
shm_subMatrixA[tx] = d_matrixA[(y-j)*aw+(x)];

__syncthreads();

for(int k = 0; k < bw; ++k) {
float b = shm_matrixB[j*bw+k];
float a = 0;
// check the out-of-bound
if ((y-j)>-1 &&(y-j)<ah&&((x)-k)>-1&&((x)-k)<aw) {
a = shm_subMatrixA[tx-k];

sum += a*b;
}
}//k loop
__syncthreads();
}//j loop
#elif defined(CHANGE1)
//modified code
for (int j=0; j<bh; j++) {
for(int k = 0; k < bw; ++k) {
float b = shm_matrixB[j*bw+k];
float a = 0;
// check the out-of-bound
if ((y-j)>-1&&(y-j)<ah&&(x-k)>-1&&(x-k)<aw) {
a = d_matrixA[(y-j)*aw+(x-k)];
sum += a*b;
}
}
} //j loop
__syncthreads();
#else
//Original Code
for (int j=0; j<bh; j++) {
for(int k = 0; k < bw; ++k) {
float b = d_matrixB[j*bw+k];
float a = 0;
// check the out-of-bound
if ((y-j)>-1&&(y-j)<ah&&(x-k)>-1&&(x-k)<aw) {
a = d_matrixA[(y-j)*aw+(x-k)];
sum += a*b;
}
}
}//j loop
#endif //CHANGES


#ifdef CHANGE4
// write data to global memory
d_matrixC[(1*y*aw)+x] = sum0;
d_matrixC[(((1*y)+1)*aw)+x] = sum1;
#elif defined(CHANGE3)
// write data to global memory
d_matrixC[(1*y*aw)+x] = sum0;
d_matrixC[(((1*y)+1)*aw)+x] = sum1;

#else
// write data to global memory
d_matrixC[y*aw+x] = sum;
#endif
}// end of func