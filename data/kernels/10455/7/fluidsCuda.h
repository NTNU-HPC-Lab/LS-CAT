#ifndef FLUIDSCUDA_H
#define FLUIDSCUDA_H

#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <sstream>

#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/max_element.h>
#include <thrust/min_element.h>

#define rowColIdx row*simWidth+col
#define REFRESH_DELAY 10 

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t result, const char *file, int line, bool abort=true)
{
	if(result != cudaSuccess)
	{
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(result), file, line);
		if (abort) exit(result);
	}
}

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

const char *vertexShaderText = 
"#version 450 \n"
"layout(location = 0) in vec3 vertexPosition;"
"layout(location = 1) in vec3 vertexColor;"
"out vec3 color;"
"void main() {"
"	color = vertexColor;"
"	gl_Position = vec4(vertexPosition, 1.0);"
"}";

const char *fragmentShaderText = 
"#version 450\n"
"in vec3 color;"
"out vec4 fragmentColor;"
"void main() {"
"	fragmentColor = vec4(color, 1.0);"
"}";


__global__ void
Obstruct(int *__restrict__ obstructed,
		 float2 *__restrict__ oldVel);

__global__ void 
colorVectorField(
	float3 *__restrict__ colors, 
	float3 *colorMap,
    float2 *__restrict__ field,
   	dim3 blocks,
	unsigned int simWidth,
    unsigned int simHeight);

void runSim(GLuint c_vbo,
			int *argc,
			char **argv,
			struct cudaGraphicsResource **vboResource, 
			unsigned int simWidth,
			unsigned int simHeight);
__global__ void 
Advect(float2 *__restrict__ positions,
	   float2 *__restrict__ oldVel, 
	   float2 *__restrict__ newVel,
	   float2 frameVel,
	   float dt,
	   float dr,
	   float4 boundaries,
	   unsigned int simWidth,
	   unsigned int simHeight,
	   unsigned int testX,
	   unsigned int testY,
	   bool test);

void runCuda(struct cudaGraphicsResource **vboResource,
			 int *__restrict__ obstructed,
			 float3 *__restrict__ colorMap,
			 float2 *__restrict__ devPositions,
			 float2 *__restrict__ devVelocities,
			 float2 *__restrict__ devVelocities2,
			 float2 *__restrict__ gradPressure,
 			 float *__restrict__ devDivVelocity,
			 float *pressure,
			 float4 boundaries,
			 float dt,
			 float dr,
			 float2 frameVel,
			 dim3 tpbColor,
			 dim3 tpbLattice,
			 dim3 blocks,
			 unsigned int simWidth,
			 unsigned int simHeight,
			 unsigned int testX,
			 unsigned int testY,
			 bool test);

__global__ void
Divergence(float2 *__restrict__ newVel, 
		   float *__restrict__ divVel,
		   float dr,
		   float2 frameVel,
		   unsigned int simWidth);

__global__ void
Gradient(float *__restrict__ field,
		 float2 *__restrict__ gradient,
		 float2 frameVel,
		 float dr,
		 unsigned int simWidth);

__global__ void
Projection(float2 *__restrict__ newVel,
		   float2 *__restrict__ gradPressure,
		   unsigned int simWidth);

__global__ void 
updateVel(float2 *__restrict__ oldVel,
		  float2 *__restrict__ newVel,
		  unsigned int simWidth);

__device__ float2
BiLinInterp(float2 pos,
 		 	 float2 TLVel, float2 TLPos,
			 float2 BLVel, float2 BLPos,
			 float2 BRVel, float2 BRPos,
			 float2 TRVel, float2 TRPos,
			 float dr);

__device__ float2
LinInterp(float2 pos,
			 float2 LVel, float2 LPos,
			 float2 RVel, float2 RPos,
			 float dr);

__global__ void
PressureJacobi(float* divVel,
			   float* Pressure,
			   float dr, 
			   unsigned int simWidth,
			   unsigned int simHeight);

__global__ void
DiffusionJacobi(float2 *__restrict__ positions,
				float2 *__restrict__ oldVel,
				float2 *__restrict__ newVel,
				float dt,
				float dr,
				float viscosity,
				unsigned int simWidth,
				unsigned int simHeight);


__device__ float2
JacobiFieldInstance(float2 Top, 
				 	float2 Left,
				    float2 Bot,
			   	    float2 Right,
			   	    float Alpha,
			  	    float2 Val);

__device__ float
JacobiScalarInstance(float Top,
					 float Left,
					 float Bot,
					 float Right,
					 float Alpha,
					 float Val);




void glInitShaders(const char *vertexShaderText,
			 	   const char *fragmentShaderText,
				   GLuint shaderProgram);

void initThreadDimensions(unsigned int simWidth,
						  unsigned int simHeight,
						  dim3 &tpb,
						  dim3 &blocks);
#endif
