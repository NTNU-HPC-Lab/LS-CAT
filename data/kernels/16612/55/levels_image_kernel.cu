#include "includes.h"
__device__ float3 hsv_to_rgb_kernel(float3 hsv)
{
float h = hsv.x;
float s = hsv.y;
float v = hsv.z;

float r, g, b;
float f, p, q, t;

if (s == 0) {
r = g = b = v;
} else {
int index = (int) floorf(h);
f = h - index;
p = v*(1-s);
q = v*(1-s*f);
t = v*(1-s*(1-f));
if(index == 0){
r = v; g = t; b = p;
} else if(index == 1){
r = q; g = v; b = p;
} else if(index == 2){
r = p; g = v; b = t;
} else if(index == 3){
r = p; g = q; b = v;
} else if(index == 4){
r = t; g = p; b = v;
} else {
r = v; g = p; b = q;
}
}
r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
return make_float3(r, g, b);
}
__device__ float3 rgb_to_hsv_kernel(float3 rgb)
{
float r = rgb.x;
float g = rgb.y;
float b = rgb.z;

float h, s, v;
float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
float delta = max - min;
v = max;
if(max == 0){
s = 0;
h = -1;
}else{
s = delta/max;
if(r == max){
h = (g - b) / delta;
} else if (g == max) {
h = 2 + (b - r) / delta;
} else {
h = 4 + (r - g) / delta;
}
if (h < 0) h += 6;
}
return make_float3(h, s, v);
}
__global__ void levels_image_kernel(float *image, float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)
{
int size = batch * w * h;
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(id >= size) return;
int x = id % w;
id /= w;
int y = id % h;
id /= h;
float rshift = rand[0];
float gshift = rand[1];
float bshift = rand[2];
float r0 = rand[8*id + 0];
float r1 = rand[8*id + 1];
float r2 = rand[8*id + 2];
float r3 = rand[8*id + 3];

saturation = r0*(saturation - 1) + 1;
saturation = (r1 > .5f) ? 1.f/saturation : saturation;
exposure = r2*(exposure - 1) + 1;
exposure = (r3 > .5f) ? 1.f/exposure : exposure;

size_t offset = id * h * w * 3;
image += offset;
float r = image[x + w*(y + h*0)];
float g = image[x + w*(y + h*1)];
float b = image[x + w*(y + h*2)];
float3 rgb = make_float3(r,g,b);
if(train){
float3 hsv = rgb_to_hsv_kernel(rgb);
hsv.y *= saturation;
hsv.z *= exposure;
rgb = hsv_to_rgb_kernel(hsv);
} else {
shift = 0;
}
image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5f)*shift;
image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5f)*shift;
image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5f)*shift;
}