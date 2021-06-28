#include "includes.h"
__global__ void kernel(float *id, float *od, int w, int h, int depth)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
const int dataTotalSize   = w * h * depth;
const int radius		  = 2;
const int filter_size	  = 2*radius + 1;
const int sW			  = 6;				/* sW == 2 * filter_radius + blockDim.x (or same as 2 * filter_radius + blockDim.y) */
/* boarder do not concerned */
if(x >= w || y >= h || z >= depth)
return;
else
{
//global defined
int idx = z*w*h+y*w+x;

//3d grid(blocks) 2d block(threads)
int threadsPerBlock = blockDim.x * blockDim.y;
int blockId		    = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId	    = (blockId * threadsPerBlock)
+ (threadIdx.y * blockDim.x) + threadIdx.x;
int g_Idx			= threadId;

//2d shared memory working
__shared__ unsigned char smem[sW][sW];
int s_Idx = threadIdx.x + (threadIdx.y * sW);
int s_IdxY = s_Idx / sW;
int s_IdxX = s_Idx % sW;

//Here: definition error, need edit, haven't finished yet.
//int g_IdxY = s_IdxY + (blockIdx.y * blockDim.y);
//int g_IdxX = s_IdxX + (blockIdx.x * blockDim.x);
//int g_Idx  = g_IdxX + (g_IdxY * w);

//32 threads working together per warp
if(s_IdxY < sW && s_IdxX < sW)	//Here: boarder concerned error, need edit
{
if(x >= 0 && y < w && y >= 0 && y < h && z >= 0 && z < depth )	//Here: boarder concerned error, need edit
smem[s_IdxY][s_IdxX] = id[g_Idx];
else
smem[s_IdxY][s_IdxX] = 0;
__syncthreads();
}

/*compute the sum using shared memory*/
float avg = 0.0;
for (int i = -radius; i <= radius; i++){
if(s_IdxY + i < 0 /*|| g_IdxY > h*/ )			//Here: boarder concerned error, need edit
avg += 0.0;
else
avg += smem[s_IdxY+i][s_IdxX];
}

/*register to global, by now thread*/
avg /= filter_size;
if(idx < dataTotalSize)
od[idx] = avg;
}
}