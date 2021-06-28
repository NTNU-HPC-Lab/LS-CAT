#include "includes.h"
__global__ void update_cluster(int *cluster, float *centroid, float *B_c, float *G_c, float *R_c, int size_image, int n_threads, int k,  float *Bdata, float *Gdata, float *Rdata, float *nValue)
{

unsigned int tid = threadIdx.x;

int size_per_thread = int(size_image/n_threads);
int start = tid*size_per_thread;
int end = start + size_per_thread;

float count = 0;
float B = 0;
float G = 0;
float R = 0;

if (tid >=size_image){ return; }

if (tid==n_threads-1)
{
start = (n_threads-1)*size_per_thread;
end = size_image;
}
for(int j = start; j < end; j++)
{
if(cluster[j] == k)
{
B = B + (B_c[j]);
G = G + (G_c[j]);
R = R + (R_c[j]);
count = count + 1;
}
}

nValue[tid] = count;
Bdata[tid] = B;
Gdata[tid] = G;
Rdata[tid] = R;

__syncthreads();

for(unsigned int s=1; s < blockDim.x; s *= 2)
{
if(tid % (2*s) == 0)
{
nValue[tid] += nValue[tid + s];
Bdata[tid] += Bdata[tid + s];
Gdata[tid] += Gdata[tid + s];
Rdata[tid] += Rdata[tid + s];
}
__syncthreads();
}

if(tid == 0)
{
if (nValue[0] != 0)
{
centroid[k*3 + 0] = Bdata[0] / nValue[0];
centroid[k*3 + 1] = Gdata[0] / nValue[0];
centroid[k*3 + 2] = Rdata[0] / nValue[0];
}
}
}