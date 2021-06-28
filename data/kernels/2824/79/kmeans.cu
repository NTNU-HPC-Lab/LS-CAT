#include "includes.h"
__global__ void kmeans (short int *input, short int*centroids, int*newcentroids, int *counter, const int n)
{
int Dim = 2;
int i = (blockIdx.x * blockDim.x + threadIdx.x)*Dim;
if ( i < n ) {
// map
int point_d0 = input[i+0];
int point_d1 = input[i+1];
int k0_d0 = point_d0 - centroids[0];
int k0_d1 = point_d1 - centroids[1];
int k1_d0 = point_d0 - centroids[2];
int k1_d1 = point_d1 - centroids[3];
k0_d0 *= k0_d0;
k0_d1 *= k0_d1;
k1_d0 *= k1_d0;
k1_d1 *= k1_d1;
// reduce sum
k0_d0 = k0_d0 + k0_d1;
k1_d0 = k1_d0 + k1_d1;
// reduce min
int k = (k0_d0 < k1_d0 ) ? 0 : 1;
// add current point to new centroids sum
atomicAdd(&(newcentroids[Dim*k]), point_d0);
atomicAdd(&(newcentroids[Dim*k+1]),point_d1);
atomicAdd(&(counter[k]),1);
} // if

}