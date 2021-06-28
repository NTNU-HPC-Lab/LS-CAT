#include "includes.h"


using namespace std;

// image size
int rows = 1224, cols = 1624;
int imgSize = rows*cols;

// iterations for stereo matching algorithm
int iteration = 1;

// disparity range
int Dmin = 1;
int Dmax = 80;
int Drange = Dmax - Dmin + 1;
//int winRadius = 9;

// device image pointer
float* dLImgPtr_f = NULL;
float* dRImgPtr_f = NULL;
size_t lPitch, rPitch;

// texture memory for stereo image pair <Type, Dim, ReadMode>
texture<float, 2, cudaReadModeElementType> lTex;
texture<float, 2, cudaReadModeElementType> rTex;

// timing arrays
const int nt = 2;
double start[nt], end[nt];
double random_start[nt], random_end[nt];
double main_start[nt], main_end[nt];


// evaluate window-based disimilarity
__device__ float evaluateCost(float u, float v, float matchIdx, int cols, int rows, int winRadius)
{
float cost = 0.0f;

for(int h=-winRadius; h<=winRadius; h++)
{
for(int w=-winRadius; w<=winRadius; w++)
{
cost += fabsf(tex2D(lTex, matchIdx+ w/(float)cols, v+h/(float)rows)
- tex2D(rTex, u+w/(float)cols, v+h/(float)rows));
}
}

return cost;
}
__global__ void stereoMatching(float* dRDispPtr, float* dRPlanes, int cols, int rows, curandState* states, int iteration)
{

int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

int winRadius = 9;

// does not need to process borders
if(x>=cols-winRadius || x<winRadius || y>=rows-winRadius || y<winRadius)
return;

float u = x/(float)cols;
float v = y/(float)rows;

int idx = y*cols +x;

// if 1st iteration, enforce planes to be fronto-parallel
if(iteration != 0)
{
// x of a unit normal vector
dRPlanes[idx*3] = 0.0f;
// y
dRPlanes[idx*3+1] = 0.0f;
// z
dRPlanes[idx*3+2] = 1.0f;
}

// evaluate disparity of current pixel
float min_cost = 0.0f;
float cost = 0.0f;
float tmp_disp = dRDispPtr[idx];
float matchIdx = u + tmp_disp*80.0f/(float)cols;

min_cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

// evaluate disparity of left neighbor
cost = 0.0f;
tmp_disp = dRDispPtr[idx-1];
matchIdx = u + tmp_disp*80.0f/(float)cols;

cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);
// update current disparity if lower cost from neighbor's
if(cost < min_cost)
{
min_cost = cost;
dRDispPtr[idx] = tmp_disp;
}

// evaluate disparity of upper neighbor
cost = 0.0f;
tmp_disp =  dRDispPtr[idx-cols];
matchIdx = u + tmp_disp*80.0f/(float)cols;

cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

if(cost < min_cost)
{
min_cost = cost;
dRDispPtr[idx] = tmp_disp;
}

// evaluate another valid random disparitiy (within border) in case it is trapped at a local minima
matchIdx= -1.0f;

while(matchIdx <(float)winRadius/cols || matchIdx >=(float)(cols-winRadius)/cols )
{
tmp_disp = curand_uniform(&states[idx]);

matchIdx = u + tmp_disp*80.0f/(float)cols;
}

cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

if(cost<min_cost)
{
min_cost = cost;
dRDispPtr[idx] = tmp_disp;
}

return;
}