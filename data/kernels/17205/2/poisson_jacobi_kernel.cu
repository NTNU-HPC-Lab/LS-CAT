#include "includes.h"
using namespace std;
#define ITERATIONS 40000


enum pixel_position {INSIDE_MASK, BOUNDRY, OUTSIDE};

__global__ void poisson_jacobi_kernel(float *targetimg, float *outimg, int *boundary_array,int c, int w, int h, int boundBoxMinX, int boundBoxMaxX, int boundBoxMinY, int boundBoxMaxY){

int x = threadIdx.x + blockIdx.x * blockDim.x + boundBoxMinX;
int y = threadIdx.y + blockIdx.y * blockDim.y + boundBoxMinY;
for(int channel = 0; channel < c; channel++){
int id = x + y*w + channel * w * h;
int idx_nextX = x+1 + w*y +w*h*channel;
int idx_prevX = x-1 + w*y + w*h*channel;
int idx_nextY = x + w*(y+1) +w*h*channel;
int idx_prevY = x + w*(y-1) +w*h*channel;
//printf("id: %d, idx_nextX: %d, idx_prevX: %d, idx_nextY: %d, idx_prevY: %d\n", id, idx_nextX, idx_prevX, idx_nextY, idx_prevY);
if(boundary_array[id] == INSIDE_MASK){
double neighbor_target = targetimg[idx_nextY]+targetimg[idx_nextX]+targetimg[idx_prevX]+targetimg[idx_prevY];
double neighbor_output = outimg[idx_nextY]+outimg[idx_nextX]+outimg[idx_prevX]+outimg[idx_prevY];
outimg[id] = 0.25*(4*targetimg[id]-neighbor_target + neighbor_output);
}
}

}