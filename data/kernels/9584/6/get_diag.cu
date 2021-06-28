#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void get_diag(double *dm9, double *out, unsigned int no_qubits) {
int x = (blockIdx.x *blockDim.x) + threadIdx.x;

if (x >= (1 << no_qubits)) return;
unsigned int addr_real = 0;
for (int i = 0; i < 16; i++) {
addr_real |= (x & 1U << i) << i | (x & 1U << i) << (i + 1);
}
out[x] = dm9[addr_real];
}