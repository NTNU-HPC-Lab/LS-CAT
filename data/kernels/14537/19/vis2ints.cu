#include "includes.h"
__global__ void vis2ints(double scale, double2 *vis_in, int2* vis_out, int npts) {
for (int q=threadIdx.x+blockIdx.x*blockDim.x;
q<npts;
q+=gridDim.x*blockDim.x) {
double2 inn = vis_in[q];
inn.x *= scale;
inn.y *= scale;
int main_y = floor(inn.y);
int sub_y = floor(GCF_GRID*(inn.y-main_y));
int main_x = floor(inn.x);
int sub_x = floor(GCF_GRID*(inn.x-main_x));
vis_out[q].x = main_x*GCF_GRID+sub_x;
vis_out[q].y = main_y*GCF_GRID+sub_y;
}
}