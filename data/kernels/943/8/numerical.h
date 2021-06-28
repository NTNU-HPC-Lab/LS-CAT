/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   numerical.h
 * Author: ziqi
 *
 * Created on January 29, 2019, 7:31 PM
 */

#ifndef NUMERICAL_H
#define NUMERICAL_H

#include <iostream>
#include <vector>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "mesh.h"
#include <algorithm>
#include <vector>

#include <gsl/gsl_sf.h>
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>


#ifndef PI
#define PI 3.1415926535897932
#endif

#ifndef INTORDER
#define INTORDER 6
#endif

#ifndef IDXC0
#define IDXC0(row,col,ld) ((ld)*(col)+(row))
#endif

#ifndef IDXC1
#define IDXC1(row,col,ld) (ld*(col-1)+(row-1))
#endif

#ifndef max
#define max(a,b) (a > b ? a : b)
#endif

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

#ifndef HOST_CALL
#define HOST_CALL(x) do {\
if(x!=EXIT_SUCCESS){\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUDA_CALL
#define CUDA_CALL(x) do {\
if((x)!=cudaSuccess) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CURAND_CALL
#define CURAND_CALL(x) do {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(x) \
do {\
if((x)!=CUBLAS_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if(x==CUBLAS_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if(x==CUBLAS_STATUS_INVALID_VALUE) {\
printf("There were problems with the parameters.\n");\
}\
if(x==CUBLAS_STATUS_MAPPING_ERROR) {\
printf("There was an error accessing GPU memory.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

#ifndef CUSOLVER_CALL
#define CUSOLVER_CALL(x) \
do {\
if((x)!=CUSOLVER_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if((x)==CUSOLVER_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if((x)==CUSOLVER_STATUS_INVALID_VALUE) {\
printf("Invalid parameters were passed.\n");\
}\
if((x)==CUSOLVER_STATUS_ARCH_MISMATCH) {\
printf("Achitecture not supported.\n"); \
}\
if((x)==CUSOLVER_STATUS_INTERNAL_ERROR) {\
printf("An internal operation failed.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

#ifndef EPS
#define EPS 0.00005
#endif

#ifndef STRENGTH
#define STRENGTH 0
#endif

//air density and speed of sound
extern __constant__ float density;

extern __constant__ float speed;

//Integral points and weights
extern __constant__ float INTPNTS[INTORDER]; 

extern __constant__ float INTWGTS[INTORDER];


__host__ __device__ void printComplexMatrix(cuFloatComplex*,const int,const int,const int); 

__host__ __device__ void printFloatMatrix(float*,const int,const int,const int);

__host__ __device__ cuFloatComplex angExpf(const float);

__host__ __device__ cuFloatComplex expfc(const cuFloatComplex);

__host__ __device__ cuFloatComplex green(const float,const float);

__host__ __device__ float PsiL(const float);

__host__ __device__ float descale(const float,const float,const float);

__host__ __device__ float arrayMin(const float*,const int);

__host__ __device__ bool inObj(const bool*,const int);

std::ostream& operator<<(std::ostream&,const cuFloatComplex&);

int Test();

//class gaussQuad
class gaussQuad {
    friend std::ostream& operator<<(std::ostream&,const gaussQuad&);
private:
    float *evalPnts = NULL;
    float *wgts = NULL;
    
    int n; //order of integration
    
    int genGaussParams();
    

    
public:
    gaussQuad(): n(INTORDER) {evalPnts=new float[n];wgts=new float[n];genGaussParams();}
    gaussQuad(const int);
    gaussQuad(const gaussQuad&);
    ~gaussQuad();
    gaussQuad& operator=(const gaussQuad&);
    
    int sendToDevice();
    
};

__host__ int lsqSolver(cuFloatComplex *A_h,const int m,const int n,const int lda,
        cuFloatComplex *B_h,const int nrhs,const int ldb, cuFloatComplex *Q_h);


int wrtCplxMat(const cuFloatComplex *mat,const int numRows,const int numCols,const int ldm,
        const char *file);

int qrSolver(const cuFloatComplex *A, const int mA, const int nA, const int ldA, 
        cuFloatComplex *B, const int nB, const int ldB);
#endif /* NUMERICAL_H */

