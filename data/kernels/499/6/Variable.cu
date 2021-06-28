#pragma once
#include <stdio.h>
#include "../ErrorChecking/ErrorChecking.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariable{
	
	int* dMem;						//ptr to memory
	int domainSize;					//variable size (cardinality)

	__host__ HostVariable(int); 	//allocate memory
	__host__ int* getPtr();			//return memory ptr;
	__host__ ~HostVariable();		//deallocate
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariable::HostVariable(int dm):
	domainSize(dm){
	ErrorChecking::hostMessage("Warn::HostVariable::HostVariable::ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMem,sizeof(int)*domainSize),"HostVariable::HostVariable");
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariable::~HostVariable(){
	ErrorChecking::hostMessage("Warn::HostVariable::~HostVariable::DEALLOCATION");
	cudaFree(dMem);
}

///////////////////////////////////////////////////////////////////////

__host__ int* HostVariable::getPtr()
	{return dMem;}


///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariable{
	int ground;			//track if variable is ground
	int changed;		//track if variable was modified
	int failed;			//track if variable is in a failed state
	int domainSize;		//size of the domain

	int* domain;		//ptr to domain memory

	int fullParallel;	//choose always parallel code execution 

	__device__ DeviceVariable();			//do nothing
	__device__ DeviceVariable(int*,int); 	//initialize
	__device__ void init(int*, int);		//initialize
	__device__ void init2(int*, int);		//initialize without setting
											//assume already setted memory
	__device__ ~DeviceVariable();			//do nothing

	__device__ int assign(int);			//assign choesen variable and returns 0.
										//otherwise -1
	__device__ int undoAssign(int);		//undo assignement
	__device__ void addTo(int,int);		//increment or decrement by delta

	__device__ void checkGround();		//check if variable is in ground state and modify ground
	__device__ void checkFailed();		//check if variable is in failed state and modify failed

	__device__ void print();			//stampa with modes

};

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::DeviceVariable(){}

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::~DeviceVariable(){}

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::DeviceVariable(int* dMem, int ds):
	domainSize(ds),ground(-1),changed(-1),failed(-1),fullParallel(true),
	domain(dMem){
		for(int i = 0; i < domainSize; ++i)dMem[i]=1;
	}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::init(int* dMem, int ds){
	domainSize = ds;
	domain = dMem;
	fullParallel = true;
	ground  = -1;
	changed = -1;
	failed  = -1;

	for(int i = 0; i < domainSize; ++i)dMem[i]=1;
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::init2(int* dMem, int ds){

	domainSize = ds;
	domain = dMem;
	fullParallel = true;
	ground  = -1;
	changed = -1;
	failed  = -1;
}

///////////////////////////////////////////////////////////////////////


__device__ inline void externAssignSequential(int* domain, int size, int value){

	for(int i = 0; i < size; ++i){
		if(i != value)--domain[i];
	}

}

__device__ void externAssignParallel(int* domain, int size, int value){

	if(threadIdx.x + blockIdx.x * blockDim.x < size && 
	   threadIdx.x + blockIdx.x * blockDim.x != value)
		--domain[threadIdx.x + blockIdx.x * blockDim.x];

}

__device__ inline int DeviceVariable::assign(int value){

/*	if(value < 0 || value >= domainSize){
		ErrorChecking::deviceError("Error::Variable::assign::ASSIGNMENT OUT OF BOUND");
		return -1;
	}

	if(failed == 1){
		ErrorChecking::deviceError("Error::Variable::assign::VARIABLE ALREADY FAILED");
		return -1;
	}

	if(domain[value]<=0){
		ErrorChecking::deviceError("Error::Variable::assign::VALUE NO MORE IN DOMAIN");
		return -1;
	}

	if(ground >= 0 && value != ground){
		ErrorChecking::deviceError("Error::Variable::assign::VARIABLE NOT GROUND");
		return -1;
	}*/


	externAssignParallel(domain, domainSize, value);

	ground = value;
	return 0;

}		

///////////////////////////////////////////////////////////////////////

__device__ inline void externUndoAssignSequential(int* domain, int size, int value){

	for(int i = 0; i < size; ++i){
		if(i != value)++domain[i];
	}

}

__global__ void externUndoAssignParallel(int* domain, int size, int value){

	if(threadIdx.x + blockIdx.x * blockDim.x < size && 
	   threadIdx.x + blockIdx.x * blockDim.x != value)
		++domain[threadIdx.x + blockIdx.x * blockDim.x];

}

__device__ inline int DeviceVariable::undoAssign(int value){

/*	if(value < 0 || value >= domainSize){
		ErrorChecking::deviceError("Error::Variable::undoAssign::OUT OF BOUND");
		return -1;
	}

	if(ground == -1){
		ErrorChecking::deviceError("Error::Variable::undoAssign::VARIABLE NOT GROUND");
		return -1;
	}*/

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceVariable::undoAssign");
	externUndoAssignParallel<<<1,domainSize>>>(domain, domainSize, value);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceVariable::undoAssign");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceVariable::undoAssign");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceVariable::undoAssign");

	checkGround();

	return 0;

}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::addTo(int value, int delta){
/*	if(value < 0 || value >= domainSize){

		ErrorChecking::deviceError("Error::Variable::addTo::ADDING OUT OF BOUND");
		return;
	}*/
	
	if(domain[value] > 0 && domain[value] + delta <= 0) changed = 1;

	domain[value]+=delta;

	checkGround();
	checkFailed();
	
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::checkGround(){

	int sum = 0;
	for(int i = 0; i < domainSize; ++i){
		if(domain[i]==1){
			++sum;
			ground = i;
		}
	}
	if(sum != 1) ground = -1;

}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::checkFailed(){

	for(int i = 0; i < domainSize; ++i)
		if(domain[i]==1){
			failed = -1;
			return;
		}
	failed = 1;

}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::print(){

	for (int i = 0; i < domainSize; ++i){
		if(domain[i] == 0)
			printf("\033[31m%d\033[0m ", domain[i]);
		else if(domain[i] > 0)printf("\033[34m%d\033[0m ", domain[i]);
		else if(domain[i] < 0)printf("\033[31m%d\033[0m ", -domain[i]);
	}

	if(ground >= 0)printf(" ::: \033[32mgrd:%d\033[0m ", ground);
	else printf(" ::: grd:%d ", ground);

	if(changed == 1)printf("\033[31mchd:%d\033[0m ", changed);
	else printf("chd:%d ", changed);

	if(failed == 1)printf("\033[31mfld:%d\033[0m ", failed);
	else printf("fld:%d ", failed);

	printf("sz:%d ", domainSize);

	printf("ptr:%d\n", domain);
}

///////////////////////////////////////////////////////////////////////
