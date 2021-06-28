#ifndef FAST_H
#define FAST_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "global_var.h"

class Fast_gpu
{

public:

    u_int8_t * imglocal_Device;
    int cols,rows;

    short2* kpLoc_Device;
    int* score_Device;

    short2* kpLocNonMaxSuppression_Device;
    float* scoreNonMaxSuppression_Device;

    int maxKeypoints;
    int threshold;

    Fast_gpu(int cols, int rows,int maxKeypoints);
    static int run_calcKeypoints(u_int8_t * img,int cols,int rows, short2* kpLoc, int maxKeypoints, int* score, int threshold);
    static int run_nonmaxSuppression_gpu(const short2* kpLoc, int count, int* score,int rows,int cols, short2* loc, float* response);
    int run_calcKeypoints(u_int8_t * img, int threshold);
    int run_nonmaxSuppression(int nbKeypoints);

};

__device__                  u_int8_t tex_u(const u_int8_t * ptData,int y,int x,int step);
__device__                  int tex_i(const int * ptData,int y,int x,int step);


__device__ __forceinline__  int diffType(const int v, const int x, const int th);
__device__                  void calcMask(const uint C[4], const int v, const int th, int& mask1, int& mask2);
__device__ __forceinline__  bool isKeyPoint(int mask1, int mask2);
__device__                  int cornerScore(const uint C[4], const int v, const int threshold);

__global__                  void calcKeypoints(const u_int8_t* img,int cols ,int rows,short2* kpLoc, const unsigned int maxKeypoints, int *score, const int threshold,bool calcScore);
                            int calcKeypoints_gpu(u_int8_t * img,int cols,int rows, short2* kpLoc, int maxKeypoints, int* score, int threshold);

__global__                  void nonmaxSuppression(const short2* kpLoc, int count, const int* scoreMat,int cols,int rows,short2* locFinal, float* responseFinal);
                            int nonmaxSuppression_gpu(const short2* kpLoc, int count, int* score,int rows,int cols, short2* loc, float* response);



#endif // FAST_H

