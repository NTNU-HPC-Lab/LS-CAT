#include "includes.h"

using namespace std;

struct  pixel //to store RGB values
{
unsigned char r;
unsigned char g;
unsigned char b;
};

__device__ pixel padding(pixel* Pixel_val, int x_coord, int y_coord, int img_width, int img_height)
{	pixel Px;
Px.r=0; Px.g=0; Px.b=0;
if(x_coord< img_width && y_coord <img_height && x_coord>=0 && y_coord>=0)
{
Px=Pixel_val[y_coord*img_width+x_coord];
}
return Px;
}
__global__ void horizontal_conv(pixel* Pixel_in, pixel* Pixel_out, int img_wd, int img_ht, float* kernel, int k)
{
float tmp_r, tmp_b, tmp_g;
//horizontal convolution
//int pix_idx=blockIdx.x*blockDim.x + threadIdx.x;
//int row=(int)(pix_idx/img_wd);
//int col=pix_idx%img_wd;
size_t col=blockIdx.x*blockDim.x + threadIdx.x;
size_t row=blockIdx.y*blockDim.y + threadIdx.y;
size_t pix_idx=row*img_wd+col;

tmp_r=0, tmp_g=0, tmp_b=0;
if(row<img_ht && col<img_wd)
{
for(int l=0; l<k;l++)
{
pixel pix_val=padding(Pixel_in, col+ l-(k-1)/2, row, img_wd, img_ht);
tmp_r+=pix_val.r * kernel[l];
tmp_g+=pix_val.g * kernel[l];
tmp_b+=pix_val.b * kernel[l];
}
Pixel_out[pix_idx].r=tmp_r;
Pixel_out[pix_idx].g=tmp_g;
Pixel_out[pix_idx].b=tmp_b;
}
}