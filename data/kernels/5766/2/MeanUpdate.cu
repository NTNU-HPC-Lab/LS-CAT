#include "includes.h"
/* Start Header
***************************************************************** /
/*!
\file knn-kernel.cu
\author Koh Wen Lin
\brief
Contains the implementation for kmeans clustering on the gpu.
*/
/* End Header
*******************************************************************/
#define KMEAN_BLOCK_SIZE 32
#define KMEAN_BLOCK_SIZE_1D KMEAN_BLOCK_SIZE * KMEAN_BLOCK_SIZE


__global__ void MeanUpdate(float* dMeanIn, unsigned k, unsigned d, int* count)
{
float ooc = 1.0f / max(1, count[threadIdx.x]);
for(int i = 0; i < d; ++i)
dMeanIn[threadIdx.x * d + i] *= ooc;
}