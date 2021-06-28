#include "includes.h"
/**
* Various matrix utils using cuda
**/


/**
* Kronecker product of two matrices kernel
* input :
* a : first matrix
* nax, nay : matrix a dimensions
* b: second matrix
* nbx, nby : matrix b dimensions
* results : kronecker product of a and b
**/

__global__ void kronecker(double * a, int nax, int nay, double * b, int nbx, int nby, double * result){

// First we need to find our global threadID
int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
int resSzx = nax * nbx;
//int resSzy = nay * nby;
int idxA = floor((tPosX) / (double)nbx);
int idyA = floor((tPosY) / (double)nby);
int idxB = (tPosX) % nbx;
int idyB = (tPosY) % nby;
// Check if the indices are within range
if (idxA >= nax || idyA > nay || idxB > nbx || idyB > nby)
{
result[tPosX + tPosY * resSzx] = -1;
return;
}
// Multiply appropriate elements
result[tPosX + tPosY * resSzx] = a[idyA * nax +  idxA] * b[idyB * nbx + idxB];
}