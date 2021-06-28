#ifndef SPHCOMPUTE_H
#define SPHCOMPUTE_H
#include "KernelOption.cuh"
#include "Particle.cuh"
#include "SPHParam.cuh"
#include <math.h>
#include <glm\vec3.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thrust\scan.h>
#include <thrust\device_vector.h>
#include "Wall.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "device_atomic_functions.h"
#include "curand.h"
#include "curand_kernel.h"
#include "CustomMath.cuh"
#include "Query.cuh"

using namespace thrust;

class SPHCompute{
public:
	
	SPHCompute(SPHParam par, KernelOption kernel, Particle *liquidParticles, Wall wallin);
	
	//added 12-16-2016==================
	SPHCompute();
	void setParam(SPHParam par);
	void setKernel(KernelOption kernel);
	void setParticles(Particle *liquidParticles);
	void setGridDimension();
	void setWall(Wall wall);
	
	//==================================
	void initCudaMemoryAlloc();
	void initCudaHostToDev();
	void initCudaParticles();
	void initCudaGrid();
	void addMoreParticles(Particle liquidParticle);
	void updateParticles();
	Particle* getAllLiquidParticles();
	Wall getWall();
	
	//-----------------------------------------------
	void copyDevToHost();
	void debugGrid();
	void debugPosition();
	void debugBin();
	void debugPrefSum();
	void debugReference();
	void debugParticleIdxAndGridSorted();
	void freeParticles();
	void freeGrid();

	//getter method
	Particle *getDeviceParticles();
	Particle *getHostParticles();
	int *getDeviceGridPositionUnsorted();
	int *getHostGridPositionUnsorted();
	int *getDeviceBin();
	int *getHostBin();
	int *getDeviceParticleIdSorted();
	int *getHostParticleIdSorted();
	int *getDeviceGridSorted();
	int *getHostGridSorted();
	int getDimenX();
	int getDimenY();
	int getDimenZ();
	int getDimenSize();
private:
	KernelOption kernelConfig;
	SPHParam param;
	Particle *h_particles;
	Particle *d_particles;

	//save gridPosition of each particles
	int *d_gridPos;
	int *h_gridPos;

	//create bin for each grid 
	int *d_bin_count;
	int *h_bin_count;
	int *d_prefSum;
	int *h_prefSum;

	//Sorted particle index & it's grid
	int *d_particleIdx;
	int *d_gridSorted;
	int *h_particleIdx;
	int *h_gridSorted;
	int *d_reference;
	int *h_reference;

	int dimen_x, dimen_y, dimen_z, dimen_size;
	Wall wall;
};

__global__ void initGridPos(Particle *d_particles, int *d_gridPos, SPHParam param, int dimen_x, int dimen_y, int dimen_z, Wall wall);
__global__ void binCount(int *d_bin_count, int *d_gridPos, SPHParam param);
__global__ void binZeros(int *d_bin_count, int bin_size);
__global__ void csSortArray(SPHParam param, int *d_gridPos, int* d_prefSum, int *d_gridSorted, int *d_particleIdx);
__device__ int getCell(int dimen_x, int dimen_y, int dimen_z, int pos_x, int pos_y, int pos_z);
__device__ void getGridFromParticlePos(Particle p, float kernelSize, Wall wall, int dimen_x, int dimen_y, int dimen_z, int* pos_x, int* pos_y, int* pos_z);
__global__ void initialParticlesValuesKernel(Particle *particles, SPHParam param);
__global__ void computeDensityAndPressureGridKernel(Particle *particles, int *particleIdx, int *reference, SPHParam param, KernelOption kernel, Wall wall, int dimen_x, int dimen_y, int dimen_z, int dimen_size);
__device__ void selfGridDensity(Particle *particles, int *particleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult);
__device__ void neighbourGridDensity(Particle *particles, int *particleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult);
__device__ bool isValidGrid(int dimen_x, int dimen_y, int dimen_z, int pos_x, int pos_y, int pos_z);
__global__ void computeForcesGridKernel(Particle *particles, int *particleIdx, int *reference, SPHParam param, KernelOption kernel, Wall wall, int dimen_x, int dimen_y, int dimen_z, int dimen_size);
__device__ void selfGridForces(Particle *particles, int *particleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult);
__device__ void neighbourGridForces(Particle *particles, int *particleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult);
__global__ void computeNewPositionKernel(Particle *particles, SPHParam param);
__global__ void computeCollisionandBoundaryKernel(Particle *particles, SPHParam param, Wall wall);
__global__ void initialise_curand(curandState * state, unsigned long seed);
__device__ float generate(curandState* globalState, int ind);
__global__ void find_boundaries(const int num_keys, const int num_bucket, const int *which_bucket, int *bucket_start);
__device__ Query query_table(const int num_bucket, const int *bucket_start, const int key);
__global__ void queryDevice(const int num_bucket, const int *bucket_start, const int key);

#endif