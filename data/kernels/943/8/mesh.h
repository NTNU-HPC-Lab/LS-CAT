/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   mesh.h
 * Author: ziqi
 *
 * Created on January 29, 2019, 10:26 PM
 */

#ifndef MESH_H
#define MESH_H
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

class cartCoord2D;
class triElem;
class mesh;

//class cartCoord
class cartCoord {
    friend std::ostream& operator<<(std::ostream&,const cartCoord&);
    
    friend __host__ __device__ cartCoord numDvd(const cartCoord&,const float);
    
    friend __host__ __device__ cartCoord numMul(const float,const cartCoord&);
    
    friend __host__ __device__ float dotProd(const cartCoord&,const cartCoord&);
    
    friend __host__ __device__ cuFloatComplex green2(const float,const cartCoord,const cartCoord);
    
    friend __host__ __device__ cuFloatComplex pntSrc(const float,const float,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float Psi_L(const cartCoord);
    
    friend __host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    
    friend __host__ __device__ float trnglArea(const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ cartCoord rayPlaneInt(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord);
    
    friend __host__ __device__ bool rayTrnglInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float r(const cartCoord,const cartCoord);
    
    friend __host__ __device__ float prpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
        const cartCoord*,const triElem*,const int,bool*);
    
    friend __global__ void test(cartCoord *pnts, triElem *elems);
    
    friend __global__ void distPntPnts(const cartCoord,const cartCoord*,const int,float*);
    
    friend __host__ __device__ float prRpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ cuFloatComplex pGpn2(const float,const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float PsiL2(const cartCoord,const cartCoord);
    
    friend __host__ __device__ float pPsiLpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __device__ void g_h_c_l_nsgl(const float &k, const cartCoord &x, const cartCoord p[3], 
        cuFloatComplex gCoeff[3], cuFloatComplex hCoeff[3], float* cCoeff);
    
    friend __device__ cuFloatComplex pntElemOffset(const float k,const cartCoord x,const triElem elem,
        const cartCoord *pnts,const cuFloatComplex *surfPressure);
    
    friend class mesh;
private:
    float coords[3];

public:
    __host__ __device__ cartCoord() {coords[0]=0;coords[1]=0;coords[2]=0;}
    __host__ __device__ cartCoord(const cartCoord&);
    __host__ __device__ cartCoord(const float x,const float y,const float z) {coords[0]=x;coords[1]=y;coords[2]=z;}
    __host__ __device__ cartCoord& operator=(const cartCoord&);
    ~cartCoord() = default;
    __host__ __device__ void set(const float,const float,const float);
    __host__ __device__ cartCoord operator+(const cartCoord&) const;
    __host__ __device__ cartCoord operator-(const cartCoord&) const;
    __host__ __device__ cartCoord operator-() const;
    __host__ __device__ cartCoord operator*(const cartCoord&) const;
    __host__ __device__ void print() {printf("(%f,%f,%f)\n",coords[0],coords[1],coords[2]);}
    __host__ __device__ float nrm2() const;
    __host__ __device__ cartCoord nrmlzd();
    __host__ __device__ bool isEqual(const cartCoord) const;
    __host__ __device__ bool isLegal() const;
    __host__ __device__ bool isInsideTrngl(const cartCoord,const cartCoord,const cartCoord) const; 
    
};

std::ostream& operator<<(std::ostream&,const cartCoord&);

__host__ __device__ cartCoord numDvd(const cartCoord&,const float);

__host__ __device__ cartCoord numMul(const float,const cartCoord&);

__host__ __device__ float dotProd(const cartCoord&,const cartCoord&);

__host__ __device__ cuFloatComplex green2(const float,const cartCoord,const cartCoord);

__host__ __device__ cuFloatComplex pntSrc(const float k, const float q, const cartCoord x, 
        const cartCoord y);

__host__ __device__ float Psi_L(const cartCoord);

__host__ __device__ float trnglArea(const cartCoord,const cartCoord);

__host__ __device__ cartCoord rayPlaneInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord);

__host__ __device__ bool rayTrnglInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float r(const cartCoord,const cartCoord);

__host__ __device__ float prpn2(const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float prRpn2(const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ cuFloatComplex pGpn2(const float,const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float PsiL2(const cartCoord,const cartCoord);

__host__ __device__ float pPsiLpn2(const cartCoord,const cartCoord,const cartCoord);

__device__ cuFloatComplex pntElemOffset(const float k,const cartCoord x,const triElem elem,
        const cartCoord *pnts,const cuFloatComplex *surfPressure);

//class triElem
class triElem {
    friend std::ostream& operator<<(std::ostream&,const triElem&);
    
    friend __device__ cuFloatComplex h_l_1_nsgl(const float k, const int xIdx, const int nod1, 
        const int nod2, const int nod3, const cartCoord *pnts);
    
    friend __global__ void test(cartCoord *pnts, triElem *elems);
    
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
    const cartCoord*,const triElem*,const int,bool*);
    
    friend __global__  void elemLPnts_nsgl(const float k, const int l, const triElem *elems, const cartCoord *pnts, 
        const int numNods, const int numCHIEF, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int numSrcs, const int ldb);
    
    friend __global__ void elemsPnts_sgl(const float k, const triElem *elems, const int numElems,
        const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3);
    
    friend __global__ void updateSystem_sgl(const triElem *elems, const int numElems, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, cuFloatComplex *A, const int lda);
    
    friend void updateSystemCPU(const triElem *elems, const int numElems, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, 
        cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, 
        cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, float *cCoeffs_sgl2, float *cCoeffs_sgl3,
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);
    
    friend __device__ cuFloatComplex pntElemOffset(const float k,const cartCoord x,const triElem elem,
        const cartCoord *pnts,const cuFloatComplex *surfPressure);
    
    friend __global__ void areaTrngls(const triElem *elems, const int numElems, const cartCoord *nodes, 
        float *area);
    
    friend __global__ void atomicPntsElems_nsgl(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);
    
    friend __global__ void atomicPntsElems_g_h_c_nsgl(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);
    
    friend __global__ void atomicPntsElems_sgl(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb);
    
    friend __global__ void atomicPntsElems_nsgl_test(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb);
    
    friend __global__ void atomicPntsElems_sgl_test(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb);
    
    friend class mesh;
    
    friend class nodElems;
private:
    int nodes[3];
    cuFloatComplex bc[3];
    
public:
    __host__ __device__ triElem() {}
    __host__ __device__ triElem(const triElem&);
    __host__ __device__ ~triElem() {}
    __host__ __device__ triElem& operator=(const triElem&);
};

std::ostream& operator<<(std::ostream&,const triElem&);



//class mesh
class mesh {
    friend std::ostream& operator<<(std::ostream&,const mesh&);
    
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
    const cartCoord*,const triElem*,const int,bool*);
    
    friend int bemSystem(const mesh &m, const float k, const cartCoord *srcs, const int numSrcs,
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int ldb);
    
    friend float cmptSurfArea(const mesh &m);
    
    friend int Test();
    
    friend class nodElems;
    
private:
    cartCoord *pnts = NULL;
    cartCoord *chiefPnts = NULL;
    triElem *elems = NULL;
    int numPnts = 0;
    int numCHIEF = 0;
    int numElems = 0;
    
    cartCoord dirCHIEF;
    
    float xl=0, xu=0, yl=0, yu=0, zl=0, zu=0;
    
public:
    mesh() = default;
    mesh(const mesh&);
    ~mesh();
    int readObj(const char*);
    mesh& operator=(const mesh&);
    int findBB(const float);
    int meshCloudToGPU(cartCoord**,triElem**) const;
    int genCHIEF(const int,const float);
    void printBB();
    void printCHIEF();
    int chiefToGPU(cartCoord**);
    int meshToGPU(cartCoord**,triElem**) const;
    int getNumChief() const {return numCHIEF;}
    int getNumPnts() const {return numPnts;}
    int getNumElems() const {return numElems;}
};

std::ostream& operator<<(std::ostream&,const mesh&);

__global__ void rayTrnglsInt(const cartCoord*,const triElem*,bool*);

float cmptSurfArea(const mesh &m);

//The elements that contribute to the pressure of a node
class nodElems {
private:
    int nodNum = 0;
    int numElems = 0;
    triElem *elems = NULL;
public:
    nodElems() = default;
    nodElems(const nodElems&);
    nodElems& operator=(const nodElems&);
    ~nodElems() {delete[] elems;}
    void findElems(const mesh &m, const int nod);
};

class cartCoord2D {
    friend __host__ __device__ cartCoord2D numDvd(const cartCoord2D&,const float);
    friend __host__ __device__ cartCoord2D numMul(const float,const cartCoord2D&);
    friend __host__ __device__ float N_1(const cartCoord2D);
    friend __host__ __device__ float N_2(const cartCoord2D);
    friend __host__ __device__ float N_3(const cartCoord2D);
    friend __host__ __device__ float pN1pXi1(const cartCoord2D);
    friend __host__ __device__ float pN1pXi2(const cartCoord2D);
    friend __host__ __device__ float pN2pXi1(const cartCoord2D);
    friend __host__ __device__ float pN2pXi2(const cartCoord2D);
    friend __host__ __device__ float pN3pXi1(const cartCoord2D);
    friend __host__ __device__ float pN3pXi2(const cartCoord2D);
    friend __host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_3(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_1(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_2(const cartCoord2D);
    friend __host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    
private:
    float coords[2];
    
public:
    __host__ __device__ cartCoord2D() {coords[0]=0;coords[1]=0;}
    __host__ __device__ cartCoord2D(const cartCoord2D&);
    __host__ __device__ cartCoord2D(const float x,const float y) {coords[0]=x;coords[1]=y;}
    ~cartCoord2D() = default;
    __host__ __device__ cartCoord2D& operator=(const cartCoord2D&);
    __host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);
    __host__ __device__ void set(const float,const float);
    __host__ __device__ cartCoord2D operator+(const cartCoord2D&) const;
    __host__ __device__ cartCoord2D operator-(const cartCoord2D&) const;
    __host__ __device__ void print() {printf("(%f,%f)\n",coords[0],coords[1]);}
};

__host__ __device__ cartCoord2D numDvd(const cartCoord2D&,const float);

__host__ __device__ cartCoord2D numMul(const float,const cartCoord2D&);

__host__ __device__ float N_1(const cartCoord2D);

__host__ __device__ float N_2(const cartCoord2D);

__host__ __device__ float N_3(const cartCoord2D);

__host__ __device__ float pN1pXi1(const cartCoord2D);

__host__ __device__ float pN1pXi2(const cartCoord2D);

__host__ __device__ float pN2pXi1(const cartCoord2D);

__host__ __device__ float pN2pXi2(const cartCoord2D);

__host__ __device__ float pN3pXi1(const cartCoord2D);

__host__ __device__ float pN3pXi2(const cartCoord2D);

__host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);

__host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_3(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_1(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_2(const cartCoord2D);

__global__ void elemLPnts_nsgl(const float k, const int l, const triElem *elems, const cartCoord *pnts, 
        const int numNods, const int numCHIEF, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int numSrcs, const int ldb);

__global__ void elemsPnts_sgl(const float k, const triElem *elems, const int numElems,
        const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int numSrcs, const int ldb);

__global__ void updateSystem_sgl(const triElem *elems, const int numElems, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, cuFloatComplex *A, const int lda);

void updateSystemCPU(const triElem *elems, const int numElems, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, 
        cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, 
        cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, float *cCoeffs_sgl2, float *cCoeffs_sgl3,
        cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb);

__device__ void g_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *gCoeff1, 
        cuFloatComplex *gCoeff2, cuFloatComplex *gCoeff3);

__device__ void h_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ cuFloatComplex h_l_1_nsgl(const float k, const int xIdx, const int nod1, 
        const int nod2, const int nod3, const cartCoord *pnts);

__device__ void g_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void g_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void g_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void h_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void h_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void h_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *pCoeff1, 
        cuFloatComplex *pCoeff2, cuFloatComplex *pCoeff3);

__device__ void c_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff);

__device__ void c_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff);

__device__ void c_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff);

__device__ void c_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff);

__device__ void g_h_c_l_nsgl(const float &k, const cartCoord &x, const cartCoord p[3], 
        cuFloatComplex gCoeff[3], cuFloatComplex hCoeff[3], float* cCoeff);

__global__ void pntElemsOffset(const float k, const cartCoord x, const triElem *elems, 
        const int numElems, const cartCoord *pnts, const cuFloatComplex *surfPressure, 
        cuFloatComplex *contribs);

cuFloatComplex genExtPressure(const float k, const mesh &m, const cartCoord src,
        const cartCoord x, const cuFloatComplex *surfPressure);
#endif /* MESH_H */

