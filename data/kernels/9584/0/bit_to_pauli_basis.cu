#include "includes.h"
/*This file is part of quantumsim. (https://github.com/brianzi/quantumsim)*/
/*(c) 2016 Brian Tarasinski*/
/*Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt*/


//kernel to transform to pauli basis (up, x, y, down)
//to be run on a complete complex density matrix, once for each bit
//this operation is its own inverse (can also be used in opposite direction)
__global__ void bit_to_pauli_basis(double *complex_dm, unsigned int mask, unsigned int no_qubits) {
const int x = (blockIdx.x *blockDim.x) + threadIdx.x;
const int y = (blockIdx.y *blockDim.y) + threadIdx.y;

const double sqrt2 =  0.70710678118654752440;
//const double sqrt2 =  1;

if ((x >= (1 << no_qubits)) || (y >= (1 << no_qubits))) return;

int b_addr = ((x|mask)<<no_qubits | (y&~mask)) << 1;
int c_addr = ((x&~mask)<<no_qubits | (y|mask)) << 1;

if (x&mask && (~y&mask)){
double b = complex_dm[b_addr];
double c = complex_dm[c_addr];
complex_dm[b_addr] = (b+c)*sqrt2;
complex_dm[c_addr] = (b-c)*sqrt2;
}
if ((~x&mask) && (y&mask)){
b_addr+=1;
c_addr+=1;
double b = complex_dm[b_addr];
double c = complex_dm[c_addr];
complex_dm[b_addr] = (b+c)*sqrt2;
complex_dm[c_addr] = (b-c)*sqrt2;
}
}