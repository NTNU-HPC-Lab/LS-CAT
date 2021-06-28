#include "includes.h"
__global__ void kmeans4 (short int *input, short int*centroids, int*newcentroids, int *counter, const int n)
{
int Dim = 4;
int i = (blockIdx.x * blockDim.x + threadIdx.x)*Dim;
if ( i < n ) {
// map
int point_d0 = input[i+0];
int point_d1 = input[i+1];
int point_d2 = input[i+2];
int point_d3 = input[i+3];

int k0_d0 = point_d0 - centroids[0];
int k0_d1 = point_d1 - centroids[1];
int k0_d2 = point_d2 - centroids[2];
int k0_d3 = point_d3 - centroids[3];

int k1_d0 = point_d0 - centroids[4];
int k1_d1 = point_d1 - centroids[5];
int k1_d2 = point_d2 - centroids[6];
int k1_d3 = point_d3 - centroids[7];

int k2_d0 = point_d0 - centroids[8];
int k2_d1 = point_d1 - centroids[9];
int k2_d2 = point_d2 - centroids[10];
int k2_d3 = point_d3 - centroids[11];

int k3_d0 = point_d0 - centroids[12];
int k3_d1 = point_d1 - centroids[13];
int k3_d2 = point_d2 - centroids[14];
int k3_d3 = point_d3 - centroids[15];


k0_d0 *= k0_d0;
k0_d1 *= k0_d1;
k0_d2 *= k0_d2;
k0_d3 *= k0_d3;

k1_d0 *= k1_d0;
k1_d1 *= k1_d1;
k1_d2 *= k1_d2;
k1_d3 *= k1_d3;

k2_d0 *= k2_d0;
k2_d1 *= k2_d1;
k2_d2 *= k2_d2;
k2_d3 *= k2_d3;

k3_d0 *= k3_d0;
k3_d1 *= k3_d1;
k3_d2 *= k3_d2;
k3_d3 *= k3_d3;

// reduce sum
k0_d0 = k0_d0 + k0_d1 + k0_d2 + k0_d3;
k1_d0 = k1_d0 + k1_d1 + k1_d2 + k1_d3;
k2_d0 = k2_d0 + k2_d1 + k2_d2 + k2_d3;
k3_d0 = k3_d0 + k3_d1 + k3_d2 + k3_d3;

// reduce min
int k01 = (k0_d0 < k1_d0 ) ? 0 : 1;
int mink01 = (k0_d0 < k1_d0 ) ?  k0_d0: k1_d0 ;
int k23 = (k2_d0 < k3_d0 ) ? 2 : 3;
int mink23 = (k2_d0 < k3_d0 ) ?  k2_d0: k3_d0 ;

int k = (mink01 < mink23) ? k01 : k23;

// add current point to new centroids sum

newcentroids[Dim*k] += point_d0;
newcentroids[Dim*k+1] +=point_d1;
newcentroids[Dim*k+2] += point_d2;
newcentroids[Dim*k+3] += point_d3;
counter[k]++;

} // if

}