#include "includes.h"


#define MAXN 8000  /* Max value of N */
int N;  /* Matrix Dimension*/
int numThreads;  /* Number of Threads */

/*Random*/
#define randm() 4|2[uid]&3

/*CUDA Function for calculating mean column-wise and then reducing each column's totals*/
/*This Function will be called Number of blocks times*/

/* returns a seed for srand based on the time */
__global__ void Mean_SD_Norm(float* input,float* output ,float* mean_out,float* sd_out, int dim1, int numThread,int eval_ceil)
{
extern __shared__ float mean[];//shared 1D-matrix for storing temporary results for mean of each threads
extern __shared__ float sd[];//shared 1D-matrix for storing temporary results for sd of each threads
__shared__ float meansum;//shared 1D-matrix for storing mean total of each threads
__shared__ float sdsum;//shared 1D-matrix for storing SD total of each threads

int idx_x = blockIdx.x * blockDim.x + threadIdx.x;//Getting Thread X Index for Particular Block
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;//Getting Thread Y Index for Particular Block
int eva_block,index;

unsigned int thread_id = threadIdx.y;//Getting Id of thread
unsigned int j = idx_y * dim1 + idx_x;//calculating index for input matrix

__syncthreads();//waiting for all threads

mean[thread_id]=input[j];//Assigned each column element of matrix to each thread

/*If Dimension is more than Threads then reduce the remaining elements to assigned elements*/
for(int i=0;i<dim1;i+=numThread)
{
index=dim1*(numThread+thread_id+i);//calculating index of remaining element
eva_block=index+blockIdx.x;
if(eva_block < dim1*dim1)
{
mean[thread_id]+=input[index];
}
}

/*Reducing sum of each thread to final block sum*/
if(thread_id==0)
{
for(int i=0;i<numThread;i++)
{
meansum+=mean[thread_id+i];
}
mean_out[blockIdx.x]=meansum/dim1;//Mean of block
}

__syncthreads();
sd[thread_id] = powf(input[j] - mean_out[blockIdx.x], 2.0);//evaluating SD for each thread for particular block


/*If Dimension is more than Threads then reduce the remaining elements to assigned elements*/
for(int i=0;i<dim1;i+=numThread)
{
index=dim1*(numThread+thread_id+i);
eva_block=index+blockIdx.x;
if(eva_block < dim1*dim1)
{
sd[thread_id]+=powf(input[index] - mean_out[blockIdx.x], 2.0);
}
}

/*Reducing SD Sum of each thread to final block SD sum*/
if(thread_id==0)
{
sdsum=0;
for(int i=0;i<numThread;i++)
{
sdsum+=sd[thread_id+i];//calculating index of remaining element
}
sd_out[blockIdx.x]=sdsum/dim1;//SD of block
}

__syncthreads();//waiting for threads

/*Normalization of each block data on basis of mean and sd of each block*/
output[blockIdx.x*dim1+thread_id] = (input[thread_id+blockIdx.x*dim1] - mean_out[blockIdx.x]) / sd_out[blockIdx.x];

/*Reducing Normalized Sum for remaining elements*/
for(int i=0;i<eval_ceil;i++){
if((numThread+thread_id)+blockIdx.x*dim1 < dim1*dim1)
{
output[(numThread+thread_id)+blockIdx.x*dim1] = (input[(numThread+thread_id)+blockIdx.x*dim1] - mean_out[blockIdx.x])/sd_out[blockIdx.x];//Normalizing the Matrix Indexes
}
}
}