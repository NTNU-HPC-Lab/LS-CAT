#include "includes.h"
__global__ void InitLabelKernel (double *Label, double xp, double yp, double rhill, double *Rmed, int nrad, int nsec)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
double distance, angle, x, y;
angle = (double)j / (double)nsec*2.0*PI;
x = Rmed[i] * cos(angle);
y = Rmed[i] * sin(angle);
distance = sqrt((x - xp) * (x - xp) + (y - yp)*(y -yp));

if (distance < rhill) Label[i*nsec + j] = 1.0;
else Label[i*nsec + j] = 0.0;

}
}