#include "includes.h"

#define THREADS 256
#define BLOCKS 32
#define NUM THREADS*BLOCKS

int seed_var =1239;

__device__ void swap(int *xp, int *yp)
{
int temp = *xp;
*xp = *yp;
*yp = temp;
}
__global__ void bitonic_sort_step(int *d_pr, int *d_bt, int j, int k)
{
int i, ixj; /* Sorting partners: i and ixj */
i = threadIdx.x + blockDim.x * blockIdx.x;
ixj = i^j;

/* The threads with the lowest ids sort the array. */
if ((ixj)>i)
{
if ((i&k)==0)
{
/* Sort ascending */
if (d_pr[i]>d_pr[ixj])
{
/* exchange(i,ixj); */
swap(&d_pr[i],&d_pr[ixj]);
swap(&d_bt[i],&d_bt[ixj]);
}
}
if ((i&k)!=0)
{
/* Sort descending */
if (d_pr[i]<d_pr[ixj])
{
/* exchange(i,ixj); */
swap(&d_pr[i], &d_pr[ixj]);
swap(&d_bt[i], &d_bt[ixj]);
}
}
}
}