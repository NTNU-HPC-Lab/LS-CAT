#include "includes.h"
/* ==================================================================
Programmers:
Kevin Wagner
Elijah Malaby
John Casey

Omptimizing SDH histograms for input larger then global memory
==================================================================
*/



#define BOX_SIZE 23000 /* size of the data box on one dimension */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
float x_pos;
float y_pos;
float z_pos;
} atom;

unsigned long long * histogram;		/* list of all buckets in the histogram */
unsigned long long  PDH_acnt;	/* total number of data points */
int block_size;		/* Number of threads per block */
int num_buckets;	/* total number of buckets in the histogram */
float   PDH_res;	/* value of w */
atom * atom_list;	/* list of all data points */
unsigned long long * histogram_GPU;
unsigned long long * temp_histogram_GPU;
atom * atom_list_GPU;


__device__ void block_to_block (atom * block_a, atom * block_b, int b_length, unsigned long long * histogram, float resolution) {
atom me = block_a[threadIdx.x];
for(int i = 0; i < b_length; i++)
atomicAdd(&(histogram[(int)(sqrt((me.x_pos - block_b[i].x_pos) * (me.x_pos - block_b[i].x_pos) +
(me.y_pos - block_b[i].y_pos) * (me.y_pos - block_b[i].y_pos) +
(me.z_pos - block_b[i].z_pos) * (me.z_pos - block_b[i].z_pos)) / resolution)]),
1);
}
__global__ void GPUKernelFunction (unsigned long long PDH_acnt, float PDH_res, atom * atom_list_GPU, unsigned long long * histogram_GPU, int num_buckets) {

extern __shared__ unsigned long long SHist[];
/* assign register values */
int i, h_pos;
float dist;
atom * my_block = &atom_list_GPU[blockIdx.x * blockDim.x];
atom temp_atom_1 = my_block[threadIdx.x];

for(h_pos=threadIdx.x; h_pos < num_buckets; h_pos+=blockDim.x)
SHist[h_pos] = 0;

__syncthreads();

/* loop through all points in atom list calculating distance from current point to all further points */
for (i = threadIdx.x + 1; i < blockDim.x && i+blockIdx.x*blockDim.x < PDH_acnt; i++)
{
atom temp_atom_2 = my_block[i];
dist = sqrt((temp_atom_1.x_pos - temp_atom_2.x_pos) * (temp_atom_1.x_pos - temp_atom_2.x_pos) +
(temp_atom_1.y_pos - temp_atom_2.y_pos) * (temp_atom_1.y_pos - temp_atom_2.y_pos) +
(temp_atom_1.z_pos - temp_atom_2.z_pos) * (temp_atom_1.z_pos - temp_atom_2.z_pos));
h_pos = (int)(dist / PDH_res);
atomicAdd(&(SHist[h_pos]), 1);
}
__syncthreads();
for(i=blockIdx.x+1; i < gridDim.x-1; i++)
block_to_block(my_block,
&atom_list_GPU[i*blockDim.x],
blockDim.x,
SHist,
PDH_res);
block_to_block(my_block,
&atom_list_GPU[i*blockDim.x],
PDH_acnt-i*blockDim.x, // Last block may be small
SHist,
PDH_res);
__syncthreads();
for(h_pos = threadIdx.x; h_pos < num_buckets; h_pos += blockDim.x)
*(histogram_GPU+(num_buckets*blockIdx.x)+h_pos) += SHist[h_pos];
}