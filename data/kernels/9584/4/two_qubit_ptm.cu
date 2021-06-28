#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void two_qubit_ptm(double *dm, double *ptm_g, unsigned int bit0, unsigned int bit1, unsigned int no_qubits) {
const unsigned int x = threadIdx.x;
const unsigned int high_x = blockIdx.x * blockDim.x;



extern __shared__ double ptm[];
double *data = &ptm[256]; //need blockDim.x double floats

// the lowest to bits of x are used to address bit0, the next two are used to address bit1
// global address = <- pos =
// aaaxxbbbbyycccc  <- aaabbbbccccxxyy

int higher_bit = max(bit0, bit1);
int lower_bit = min(bit0, bit1);
int high_mask = ~ ( (1 << (2*higher_bit+2)) - 1 ); //a mask (of pos)
int mid_mask = (~ ( (1 << (2*lower_bit + 4)) - 1)) & (~high_mask);  //b mask
int low_mask  = ~(high_mask | mid_mask) & (~0xf);  //c mask

int pos = high_x | x;
int global_from =
(pos & high_mask)
| ((pos & mid_mask) >> 2)
| ((pos & low_mask) >> 4)
| ((pos & 0x3) << (2 * bit0))
| (((pos & 0xc) >>2)  << (2 * bit1));

//fetch ptm to shared memmory
//need to fetch several values per thread if blockDim.x is less than 256 (only for small dms...)
for(int i=0; i < 256; i+=blockDim.x) {
if(i+x < 256) {
ptm[i+x] = ptm_g[i+x];
}
}
if (high_x + x >= (1 << (2*no_qubits))) return;


//fetch data block to shared memory
data[x] = dm[global_from];
__syncthreads();

unsigned int row = x & 0xf;
unsigned int idx = x & ~0xf;

double acc=0;
for(int i=0; i<16; i++) {
acc += ptm[16*row + i]*data[idx+i];
}


__syncthreads();
dm[global_from] = acc;

}