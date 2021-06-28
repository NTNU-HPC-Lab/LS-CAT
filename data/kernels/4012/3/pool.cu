#include "includes.h"
__global__ void pool(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count)
{
// process image
int offset = (blockIdx.x * blockDim.x + threadIdx.x)*4;

for (int i = offset; i < (width*height); i+=(thread_count*4) )
{
int x = i % (width * 2) * 2;
int y = i / (width * 2);
int p1 = 8 * width * y + x;
int p2 = 8 * width * y + x + 4;
int p3 = 8 * width * y + x + 4 *  width;
int p4 = 8 * width * y + x + 4 * width + 4;

unsigned r[] = { image[p1],   image[p2],   image[p3],   image[p4] };
unsigned g[] = { image[p1+1], image[p2+1], image[p3+1], image[p4+1] };
unsigned b[] = { image[p1+2], image[p2+2], image[p3+2], image[p4+2] };
unsigned a[] = { image[p1+3], image[p2+3], image[p3+3], image[p4+3] };

int rMax = r[0];
int gMax = g[0];
int bMax = b[0];
int aMax = a[0];

for (int j = 1; j < 4; j++ )
{
if (r[j] > rMax) rMax = r[j];
if (g[j] > gMax) gMax = g[j];
if (b[j] > bMax) bMax = b[j];
if (a[j] > aMax) aMax = a[j];

}
new_image[i] = rMax;
new_image[i+1] = gMax;
new_image[i+2] = bMax;
new_image[i+3] = aMax;


}
}