#include "includes.h"
__global__ void gaussian_filter(unsigned *in, unsigned *out, int width, int height){
__shared__ int cikti;
cikti = 0;

__syncthreads();

cikti += in[blockIdx.y*width*2 + blockIdx.x*2 + threadIdx.y*width + threadIdx.x];

__syncthreads();

out[blockIdx.y*width/2 + blockIdx.x] = cikti; // ciktiyi bir sayiya boldugumde garip bir sekilde resim karariyor???(oysa 4 sayiyi topluyoruz, neden ort almiyoruz???)

}