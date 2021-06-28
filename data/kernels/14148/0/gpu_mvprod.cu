#include "includes.h"
/* Copyright (C) 2012  Ward Poelmans

This file is part of Hubbard-GPU.

Hubbard-GPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hubbard-GPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Hubbard-GPU.  If not, see <http://www.gnu.org/licenses/>.
*/


// number of threads in a block (must be multiple of 32)
#define NUMTHREADS 128

// the maximum size of the grid
#define GRIDSIZE 65535

// Helper macro to check CUDA return values
__global__ void gpu_mvprod(double *x, double *y, double alpha, int NumUp, int NumDown, int dim, double *Umat, double *Down_data,unsigned int *Down_ind, int size_Down, double *Up_data, unsigned int *Up_ind, int size_Up, int rows_shared)
{
int index = threadIdx.x + blockDim.x * blockIdx.x + blockIdx.y * blockDim.x * gridDim.x;

if(index < dim)
{
double result = Umat[index] * x[index];

int sv = index / NumDown; //__fdividef(index,NumDown);
int id = index % NumDown; // index - sv*NumDown;

extern __shared__ double shared[];

unsigned int *shared_ind = (unsigned int *) &shared[size_Up * rows_shared];

int s_sv = (blockDim.x * blockIdx.x + blockIdx.y * blockDim.x * gridDim.x)/NumDown;

if(threadIdx.x < rows_shared && (s_sv + threadIdx.x) < NumUp)
for(int i=0;i<size_Up;i++)
{
shared[i*rows_shared+threadIdx.x] = Up_data[s_sv + threadIdx.x + i*NumUp];

shared_ind[i*rows_shared+threadIdx.x] = Up_ind[s_sv + threadIdx.x + i*NumUp];
}

__syncthreads();

for(int i=0;i<size_Up;i++)
// result += Up_data[sv+i*NumUp] * x[id + NumDown*Up_ind[sv+i*NumUp]];
result += shared[sv-s_sv+i*rows_shared] * x[id + NumDown*shared_ind[sv-s_sv+i*rows_shared]];

for(int i=0;i<size_Down;i++)
result += Down_data[id+i*NumDown] * x[sv*NumDown + Down_ind[id+i*NumDown]];

y[index] = alpha * y[index] + result;
}
}