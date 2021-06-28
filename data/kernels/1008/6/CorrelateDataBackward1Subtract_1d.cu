#include "includes.h"

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

// == Dimension rearrangement Kernel

__global__ void CorrelateDataBackward1Subtract_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2, int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size, const float *bottom0, const float *bottom1, float *bottom1diff, const float *topdiff)
{
CUDA_KERNEL_LOOP(index, nthreads) {
int l = index % bottomwidth + pad_size; //w-pos
int m = (index / bottomwidth) % bottomheight; //h-pos
int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

// round_off is a trick to enable integer division with ceil, even for negative numbers
// We use a large offset, for the inner part not to become negative.
const int round_off = ROUND_OFF;
const int round_off_s1 = stride1 * round_off;

float sum = 0;
{
for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

int s2o = stride2 * o;

//Get X,Y ranges and clamp
// We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

// Same here:
int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
{
xmin = max(0,xmin);
xmax = min(topwidth-1,xmax);

ymin = max(0,ymin);
ymax = min(topheight-1,ymax);

// Get bottom0 data:
int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
float bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
float bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
float sign = (bot0tmp >= bot1tmp) ? float(-1.0) : float(1.0);

// Index offset for topdiff in following loops:
int op = (o-x_shift); // index [o,p]
int idxOpOffset = (item * topchannels + op);

for(int y = ymin; y <= ymax; y++) {
for(int x = xmin; x <= xmax; x++) {
int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
sum += topdiff[idxtopdiff] * sign;
}
}
}
}
}
const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
}

}