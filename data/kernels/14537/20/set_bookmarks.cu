#include "includes.h"
__global__ void set_bookmarks(int2* vis_in, int npts, int blocksize, int blockgrid, int* bookmarks) {
for (int q=threadIdx.x+blockIdx.x*blockDim.x;q<=npts;q+=gridDim.x*blockDim.x) {
int2 this_vis = vis_in[q];
int2 last_vis = vis_in[q-1];
int main_x = this_vis.x/GCF_GRID/blocksize;
int main_x_last = last_vis.x/GCF_GRID/blocksize;
int main_y = this_vis.y/GCF_GRID/blocksize;
int main_y_last = last_vis.y/GCF_GRID/blocksize;
if (0==q) {
main_y_last=0;
main_x_last=-1;
}
if (npts==q) main_x = main_y = blockgrid;
if (main_x != main_x_last || main_y != main_y_last)  {
for (int z=main_y_last*blockgrid+main_x_last+1;
z<=main_y*blockgrid+main_x; z++) {
bookmarks[z] = q;
}
}
}
}