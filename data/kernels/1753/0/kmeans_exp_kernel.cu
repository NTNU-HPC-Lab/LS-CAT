#include "includes.h"
// Includes, system

// prototype function
//int rnd_asg(unsigned short int*, unsigned int*, int, int);
//void criterion_part(float*, unsigned short int*, unsigned int*, float*, float*, float*, float*, int, int, int);

// ERROR system
#define EXIT_OK (0)
#define ERROR_HOST_MEM (1)
#define ERROR_DEVICE_MEM (2)
#define ERROR_DEVICE (3)
#define ERROR_INIT (4)
#define ERROR_EMPTY (5)
#define ERROR_SETDEVICE (6)
#define EXIT_DONE (255)

// kernel to calculate the exp

__global__ void kmeans_exp_kernel(float* DIST, float pw)
{
register int idx = blockIdx.x * blockDim.x + threadIdx.x;
register float arg = DIST[idx] * pw;
if (arg < -70) arg = -70;
DIST[idx] = exp(arg);
}