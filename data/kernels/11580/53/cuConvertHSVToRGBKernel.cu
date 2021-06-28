#include "includes.h"
__global__ void cuConvertHSVToRGBKernel(const float4* src, float4* dst, size_t stride, int width, int height, bool denormalize)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int c = y*stride + x;

if (x<width && y<height)
{
// Read
float4 in = src[c];
float H = in.x;
float S = in.y;
float V = in.z;

float4 rgb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

//    float C = V*S;

//    // Denormalize
//    if (denormalize)
//      H = H*360.0f;

//    // RGB
//    H /= 60.0f;
//    float X = C*(1.0f - abs(fmod(H, 2.0f) - 1.0f));


//    if (H >= 0.0f)
//    {
//      if (H < 1.0f)
//        rgb = make_float4(C, X, 0.0f, 0.0f);
//      else if (H < 2.0f)
//        rgb = make_float4(X, C, 0.0f, 0.0f);
//      else if (H < 3.0f)
//        rgb = make_float4(0.0f, C, X, 0.0f);
//      else if (H < 4.0f)
//        rgb = make_float4(0.0f, X, C, 0.0f);
//      else if (H < 5.0f)
//        rgb = make_float4(X, 0.0f, C, 0.0f);
//      else if (H <= 6.0f)
//        rgb = make_float4(C, 0.0f, X, 0.0f);
//    }

//    float m = V-C;
//    rgb += m;

if (S == 0)
{
rgb = make_float4(V, V, V, in.w);
dst[c] = rgb;
return;
}

H /= 60.0f;
int i = floor(H);
float f = H-i;
float p = V*(1.0f - S);
float q = V*(1.0f - S*f);
float t = V*(1.0f - S*(1.0f-f));

if (i == 0)
rgb = make_float4(V, t, p, in.w);
else if (i == 1)
rgb = make_float4(q, V, p, in.w);
else if (i == 2)
rgb = make_float4(p, V, t, in.w);
else if (i == 3)
rgb = make_float4(p, q, V, in.w);
else if (i == 4)
rgb = make_float4(t, p, V, in.w);
else if (i == 5)
rgb = make_float4(V, p, q, in.w);



// Write Back
rgb.w = in.w;
dst[c] = rgb;
}
}