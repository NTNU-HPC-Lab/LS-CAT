#include "includes.h"
// customDllFunctions.cu

//////////////////////////
// Template to write .dlls
//////////////////////////

/* Include the following directories for the program to run appropriately:
///////////////////////
in the VC++ directories:

$(VC_IncludePath);
$(WindowsSDK_IncludePath);
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.0\common\inc;
$(CUDA_INC_PATH)
C:\Program Files\National Instruments\LabVIEW 2015\cintools

////////////////////////
CUDA/C/C++ directories:
./
../../common/inc
$(CudaToolkitDir)/include

////////////////////////////////
Linker/General include libraries:
cudart.lib

//changed the target machine platform from 32 to 64 bit
*/





////////////////////////////////////////////////////////////////////////////////
// Complex operations,
////////////////////////////////////////////////////////////////////////////////


__device__ static __inline__ float cmagf(float x, float y)
{
float a, b, v, w, t;
a = fabsf(x);
b = fabsf(y);
if (a > b) {
v = a;
w = b;
}
else {
v = b;
w = a;
}
t = w / v;
t = 1.0f + t * t;
t = v * sqrtf(t);
if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
t = v + w;
}
return t;
}
__global__ void ConvertCmplx2Polar(float* inRe, float* inIm, float* mag, float* phase, int size) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = threadID; i < size; i += numThreads)
{
phase[i] = atan2f(inIm[i], inRe[i]);
mag[i] = cmagf(inIm[i], inRe[i]);
}
}