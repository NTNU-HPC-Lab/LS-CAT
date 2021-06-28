/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   atomicFuncs.h
 * Author: ziqi
 *
 * Created on February 13, 2019, 7:33 AM
 */

#ifndef ATOMICFUNCS_H
#define ATOMICFUNCS_H
#include <cuComplex.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "mesh.h"

__global__ void floatComplexAdd(cuFloatComplex *a, cuFloatComplex *b, const int num);

__global__ void add(float *loc, float *temp, const int num);

__global__ void atomicPntsElems_nsgl(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);

__global__ void atomicPntsElems_g_h_c_nsgl(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);

__global__ void atomicPntsElems_sgl(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb);

int atomicGenSystem(const float k, const triElem *elems, const int numElems, 
        const cartCoord *pnts, const int numNods, const int numCHIEF, 
        const cartCoord *srcs, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb);

__global__ void atomicPntsElems_nsgl_test(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);

__global__ void atomicPntsElems_sgl_test(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb);

int atomicGenSystem_test(const float k, const triElem *elems, const int numElems, 
        const cartCoord *pnts, const int numNods, const int numCHIEF, 
        const cartCoord *srcs, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb);

#endif /* ATOMICFUNCS_H */

