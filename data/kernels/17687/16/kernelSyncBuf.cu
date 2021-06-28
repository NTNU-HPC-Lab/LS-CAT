#include "includes.h"
__global__ void kernelSyncBuf(double *A, double *A0)
{
const int i = threadIdx.x;
const int j = threadIdx.y;
const int k = threadIdx.z;
const int N1 = blockDim.x;
const int N2 = blockDim.y;
const int N3 = blockDim.z;

const int iB = blockIdx.x;
const int jB = blockIdx.y;
const int kB = blockIdx.z;
//const int N1B = gridDim.x;	//just never used
const int N2B = gridDim.y;
const int N3B = gridDim.z;

const int iG = i + iB * N1;
const int jG = j + jB * N2;
const int kG = k + kB * N3;
//const int N1G = N1 * N1B;		//just never used
const int N2G = N2 * N2B;
const int N3G = N3 * N3B;

const int indB = k + N3 * (j + N2 * i);
const int indA = kB + N3B * (jB + N2B * iB);
const int indA0 = kG + N3G * (jG + N2G * iG);

extern __shared__ double B[];
B[indB] = A0[indA0];
__syncthreads();


int numOfElem = N1 * N2 * N3;
int step = 1;
while (numOfElem > 1)
{
if (indB % (2*step) == 0)
{
B[indB] = B[indB] + B[indB + step];
}
__syncthreads();

numOfElem /= 2;
step *= 2;

}

if (indB == 0)
{
A[indA] = B[0] / (N1 * N2 * N3);
}

}