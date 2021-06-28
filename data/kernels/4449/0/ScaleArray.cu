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


__global__ void ScaleArray(float *d_a, float alpha, int arraySize)
{
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
float temp;

for (int i = threadID; i < arraySize; i += numThreads)
{
temp = d_a[i];
d_a[i] = alpha*temp;
}
}