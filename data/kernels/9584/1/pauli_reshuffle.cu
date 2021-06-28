#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void pauli_reshuffle(double *complex_dm, double *real_dm, unsigned int no_qubits, unsigned int direction) {

const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;


//do we need imaginary part? That is the case if we have an odd number of bits for y in our adress (bit in y is 1, bit in x is 0)
unsigned int v = ~x & y;


unsigned int py = 0;
while (v) {
py += v&1;
v >>= 1;
}

py = py & 0x3;

//short version: while (v>1) { v = (v >> 1) ^ v ;}
//bit bang version
/*v ^= v >> 1;*/
/*v ^= v >> 2;*/
/*v = (v & 0x11111111U) * 0x11111111U;*/
/*v = (v >> 28) & 1;*/

const unsigned int addr_complex = (((x << no_qubits) | y) << 1) + (py&1);


//the adress in pauli basis is obtained by interleaving
unsigned int addr_real = 0;
for (int i = 0; i < 16; i++) {
addr_real |= (x & 1U << i) << i | (y & 1U << i) << (i + 1);
}


if(direction == 0) {
real_dm[addr_real] = ((py==3 || py==2)? -1 : 1)*complex_dm[addr_complex];
}
else {
complex_dm[addr_complex] = ((py==3 || py == 2)? -1 : 1)*real_dm[addr_real];
}
}