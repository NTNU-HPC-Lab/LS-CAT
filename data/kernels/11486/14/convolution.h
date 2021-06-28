#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <helper_cuda.h>
#include <assert.h>

#define kBlockDimX 16
#define kBlockDimY 16

#define CACHE_W 16
#define CACHE_H 16
#define KERNAL_RAD 8
#define KERNEL_LEN (2 * KERNAL_RAD + 1)

extern "C" void setKernel(float *h_Kernel);

// Naive approach
extern "C" void convolutionFullNaiveSepKernel(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH, int kernelR);
extern "C" void convolutionFullNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,float* d_Kernel,int  imageW,int imageH, int kernelR);

extern "C" void convolutionSeparableColumnNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR);
extern "C" void convolutionSeparableRowNaive(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR);


// Shared approach
extern "C" void convolutionSeparableColumnShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR);
extern "C" void convolutionSeparableRowShared(dim3 gridSize, dim3 blockSize, float* d_Input,float* d_Output,int  imageW,int imageH,int kernelR);

// Shared approach with unroll
extern "C" void convolutionSeparableColumnSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);
extern "C" void convolutionSeparableRowSharedUnroll(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);

// Shared approach _mul24 and unroll
extern "C" void convolutionSeparableColumnSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);
extern "C" void convolutionSeparableRowSharedMul(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);

// Shared approach multiple pixels per thread
extern "C" void convolutionSeparableColumnSharedTile(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);
extern "C" void convolutionSeparableRowSharedTile(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);

// Shared approach multiple pixels per thread with coalescence
extern "C" void convolutionSeparableColumnSharedTileCoales(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);
extern "C" void convolutionSeparableRowSharedTileCoales(dim3 gridSize, dim3 blockSize, float* d_Input, float* d_Output, int  imageW, int imageH);

#endif

