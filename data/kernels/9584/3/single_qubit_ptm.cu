#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void single_qubit_ptm(double *dm, double *ptm_g,  unsigned int bit, unsigned int no_qubits) {
const unsigned int x = threadIdx.x;
const unsigned int high_x = blockIdx.x * blockDim.x;

if (high_x + x >= (1 << (2*no_qubits))) return;

//the two lowest bits of thread id are used to index the target bit,
//      xx <- target bit
int high_mask = ~ ( (1 << (2*bit+2)) - 1 ); // 1111100000000
int low_mask  = ~high_mask & (~0x3);        // 0000011111100

int pos = high_x | x;
int global_from = (pos & high_mask) | ((pos & 0x3) << (2*bit)) | ((pos & low_mask)>>2);

extern __shared__ double ptm[];
double *data = &ptm[16]; //need blockDim.x double floats

//first fetch the transfer matrix to shared memory
if(x < 16) ptm[x] = ptm_g[x];

if(no_qubits < 2) { //what a boring situation
ptm[x+4] = ptm_g[x+4];
ptm[x+8] = ptm_g[x+8];
ptm[x+12] = ptm_g[x+12];
}

//fetch block to shared memory
data[x] = dm[global_from];
__syncthreads();

//do calculation

int row = x & 0x3;
int idx = x & ~0x3;

double acc = 0;

acc += ptm[4*row    ] * data[idx    ];
acc += ptm[4*row + 1] * data[idx + 1];
acc += ptm[4*row + 2] * data[idx + 2];
acc += ptm[4*row + 3] * data[idx + 3];

//upload back to global memory
__syncthreads();
dm[global_from] = acc;
}