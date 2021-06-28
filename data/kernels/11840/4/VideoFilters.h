/******************************************************************************
*
*            (C) Copyright 2014 The Board of Trustees of the
*                        Florida Institute of Technology
*                         All Rights Reserved
*
* Lab Video Filters
******************************************************************************/
extern "C"
void CUDA_CreateMemoryArray(int imageW,int imageH);
extern "C"
void CUDA_BindTextureToArray();
extern "C"
void CUDA_MemcpyToArray(uchar4 *src, int imageW, int imageH);
extern "C"
void CUDA_FreeArrays();

/*Image Filters*/
extern "C"
void CUDA_MeanFilter(uchar4 *Image_dev,int imageW,int imageH,dim3 grid,dim3 threads);
extern "C"
void CUDA_LaplacianFilter(float *Image_dev,int imageW,int imageH,dim3 grid,dim3 threads);
extern "C"
void CUDA_GaussianFilter(uchar4 *Image_dev,int imageW,int imageH,dim3 grid,dim3 threads);

