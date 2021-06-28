#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void dm_reduce(double *dm, unsigned int bit, double *dm0, unsigned int state, unsigned int no_qubits) {

const int addr = blockIdx.x*blockDim.x + threadIdx.x;

if(addr >= (1<< (2*no_qubits))) return;

const int low_mask = (1 << (2*bit))-1;      //0000011111
const int high_mask = (~low_mask) << 2;     //1110000000

if(((addr >> (2*bit)) & 0x3) == state*0x3) {
dm0[ (addr & low_mask) | ((addr & high_mask) >> 2) ] = dm[addr];
}
}