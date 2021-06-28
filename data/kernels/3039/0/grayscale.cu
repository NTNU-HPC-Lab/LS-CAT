#include "includes.h"
/*----------------------------------------------------------------

*

* Multiprocesadores: Cuda

* Fecha: 11-Nov-2019

* Autor: A01206747 Mariana Perez
Autor: A01205559 Roberto Nuñez

* Image = 1080 x 1920
Speedup =  33.93700 ms / 0.00250 ms  = 13.5748

*--------------------------------------------------------------*/



__global__ void grayscale(unsigned char *src, unsigned char *dest, int width, int height, int nChannels) {
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;

if(y < height && x < width) {
int pos = (y * width + x) * nChannels;

unsigned char r = (float)src[pos];
unsigned char g = (float)src[pos + 1];
unsigned char b = (float)src[pos + 2];
dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char)0.2126 * r + 0.7152 * g + 0.0722 * b;
}
}