#include <iostream>
#include <stdio.h>
using namespace std;
__global__ void initialize(float* a, float* oA, float* x, float totalSize, int n, int ghosts);
__global__ void advect(float dt, float* a, float* oA, float* x, float u, int n, int ghosts, float tmax);
__global__ void initSinusoid(float* a, float* x, float totalX, int n, int ghosts, float shift, float amp);
__global__ void initSquare(float* a, float* x, float totalX, int n, int ghosts);
__device__ void setA(int x, float init, float* a);
__device__ float linInterp(float* in);
__device__ float colellaEvenInterp(float*in);

class CFD{
public:
	CFD(int x, float size, float uIn);
	void setInitial(int x, float init);	//create point of energy at specific cell
	void step(float maxtime);	//solve for time step dt
	float* getA();
	int getDim();
private:
	int dim, ghosts;
	float* a, u, *x;
	float* d_a, *d_x, *d_oA;
	float totalX;
	const int maxThreads = 1024;
	int numBlocks;
};