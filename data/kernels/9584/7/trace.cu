#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void trace(double *diag, int bit) {
unsigned int x = threadIdx.x;
unsigned int mask = 0;

if(bit >= 0) {
mask = 1 << bit;
}

extern __shared__ double s_diag[];
s_diag[x] = diag[x];
__syncthreads();

double a;

for(unsigned int i=1; i < blockDim.x; i <<= 1) {
if(i != mask && i <= x) {
a = s_diag[x-i];

}
__syncthreads();
if(i != mask && i <= x) {
s_diag[x] += a;
}
__syncthreads();
}

__syncthreads();
//copy result back
if(x == 0) {
diag[blockIdx.x] = s_diag[blockDim.x - 1];
return;
}
if(x == 1 && bit >= 0) {
diag[blockIdx.x + 1] = s_diag[blockDim.x - 1 - mask];
return;
}
}