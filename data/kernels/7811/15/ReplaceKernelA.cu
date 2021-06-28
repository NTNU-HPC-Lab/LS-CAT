#include "includes.h"
__global__ void ReplaceKernelA(const float* p_Input, float* p_Output, int p_Width, int p_Height, float hueRangeA, float hueRangeB, float hueRangeWithRollOffA, float hueRangeWithRollOffB, float satRangeA, float satRangeB, float satRolloff, float valRangeA, float valRangeB, float valRolloff, int OutputAlpha, int DisplayAlpha, float p_Black, float p_White) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float hcoeff, scoeff, vcoeff;
float r, g, b, h, s, v;
r = p_Input[index];
g = p_Input[index + 1];
b = p_Input[index + 2];
float min = fmin(fmin(r, g), b);
float max = fmax(fmax(r, g), b);
v = max;
float delta = max - min;
if (max != 0.0f) {
s = delta / max;
} else {
s = 0.0f;
h = 0.0f;
}
if (delta == 0.0f) {
h = 0.0f;
} else if (r == max) {
h = (g - b) / delta;
} else if (g == max) {
h = 2 + (b - r) / delta;
} else {
h = 4 + (r - g) / delta;
}
h *= 1 / 6.0f;
if (h < 0.0f) {
h += 1.0f;
}
h *= 360.0f;
float h0 = hueRangeA;
float h1 = hueRangeB;
float h0mrolloff = hueRangeWithRollOffA;
float h1prolloff = hueRangeWithRollOffB;
if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) {
hcoeff = 1.0f;
} else {
float c0 = 0.0f;
float c1 = 0.0f;
if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) {
c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f :
((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);
}
if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) {
c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f :
((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);
}
hcoeff = fmax(c0, c1);
}
float s0 = satRangeA;
float s1 = satRangeB;
float s0mrolloff = s0 - satRolloff;
float s1prolloff = s1 + satRolloff;
if ( s0 <= s && s <= s1 ) {
scoeff = 1.0f;
} else if ( s0mrolloff <= s && s <= s0 ) {
scoeff = (s - s0mrolloff) / satRolloff;
} else if ( s1 <= s && s <= s1prolloff ) {
scoeff = (s1prolloff - s) / satRolloff;
} else {
scoeff = 0.0f;
}
float v0 = valRangeA;
float v1 = valRangeB;
float v0mrolloff = v0 - valRolloff;
float v1prolloff = v1 + valRolloff;
if ( (v0 <= v) && (v <= v1) ) {
vcoeff = 1.0f;
} else if ( v0mrolloff <= v && v <= v0 ) {
vcoeff = (v - v0mrolloff) / valRolloff;
} else if ( v1 <= v && v <= v1prolloff ) {
vcoeff = (v1prolloff - v) / valRolloff;
} else {
vcoeff = 0.0f;
}
float coeff = fmin(fmin(hcoeff, scoeff), vcoeff);
float A = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff :
OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ?
fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff);
if (DisplayAlpha == 0)
A = coeff;
if (p_Black > 0.0f)
A = fmax(A - (p_Black * 4.0f) * (1.0f - A), 0.0f);
if (p_White > 0.0f)
A = fmin(A * (1.0f + p_White * 4.0f), 1.0f);
p_Output[index] = h;
p_Output[index + 1] = s;
p_Output[index + 2] = v;
p_Output[index + 3] = A;
}}