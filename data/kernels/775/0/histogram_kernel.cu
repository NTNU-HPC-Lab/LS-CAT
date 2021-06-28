#include "includes.h"

#define H 	64

// Default values
int N = 10000; 		//Size
int T = 32; 		//BlockSize
int B = 4; 		//Blocks

// Host Variables
int* HostData;
int* HostHist;
int* HostTimer=NULL;

// Device Variables
int* DeviceData;
int* DeviceHist;
int* DeviceTimer=NULL;

// Timer Variables
struct timeval CPU_Time_start, CPU_Time_end;
struct timeval GPU_Time_start, GPU_Time_end;
struct timeval DeviceToHost_start, DeviceToHost_end;
struct timeval HostToDevice_start, HostToDevice_end;
struct timeval CPU_Partial_Time_start, CPU_Partial_Time_end;
struct timeval CPU_Cleanup_Time_start, CPU_Cleanup_Time_end;
struct timeval Total_Time_start, Total_Time_end;


// Function Declaration
void Cleanup(void);
void HistogramSequential(int* result, int* data, int size);

// Histogram kernel


__global__ void histogram_kernel(int* PartialHist, int* DeviceData, int dataCount,int* timer)
{
unsigned int tid = threadIdx.x;
unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int stride = blockDim.x * gridDim.x;
clock_t start_clock=0;
clock_t stop_clock=0;

if(tid==0)
{
start_clock = clock();
}

__shared__ int BlockHist[H];

extern __shared__ int hist[];

for(int h = 0; h < H; h++)
{
hist[tid * H + h]=0;
}

BlockHist[tid] = 0;
BlockHist[tid + blockDim.x] = 0;

for(int pos = gid; pos < dataCount; pos += stride)
hist[tid * H + DeviceData[pos]]++;

for(int t_hist = 0; t_hist < blockDim.x; t_hist++)
{
BlockHist[tid] += hist[t_hist * H + tid];
BlockHist[tid+blockDim.x] += hist[(t_hist * H)+(tid + blockDim.x)];
}

PartialHist[tid+(blockIdx.x * H)] = BlockHist[tid];
PartialHist[tid+(blockIdx.x * H) + blockDim.x] = BlockHist[tid + blockDim.x];

if(tid==0)
{
stop_clock = clock();
timer[blockIdx.x * 2] = start_clock;
timer[blockIdx.x * 2 + 1] = stop_clock;
}
}