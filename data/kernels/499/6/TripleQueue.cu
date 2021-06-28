#pragma once
#include <stdio.h>
#include "../ErrorChecking/ErrorChecking.cu"


//////////////////////////////////////////////////////////////////////

struct Triple{
	int var;	//variable propagated
	int val;	//value propagated
	int cs;		//constraint used for propagation
};

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostQueue{
	Triple* dMem;			//memory ptr
	int nQueen;				//number of queen
	int size;				//max size of queue
	bool dbg;				//verbose

	__host__ HostQueue(int);	//allocate memory
	__host__ Triple* getPtr();		//return memory ptr;
	__host__ ~HostQueue();			//deallocate
};

///////////////////////////////////////////////////////////////////////

__host__ HostQueue::HostQueue(int nq):size(nq*nq*3),nQueen(nq),dbg(true){

	ErrorChecking::hostMessage("Warn::HostQueue::constructor::ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMem,sizeof(Triple)*nQueen*nQueen*3),"HostQueue::HostQueue");

}

///////////////////////////////////////////////////////////////////////

__host__ HostQueue::~HostQueue(){
	ErrorChecking::hostMessage("Warn::HostQueue::destructor::DELLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(dMem),"HostQueue::~HostQueue");
}

///////////////////////////////////////////////////////////////////////

__host__ Triple* HostQueue::getPtr()
	{return dMem;}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceQueue{
	int nQueen;		//number of queen
	int count;		//number of element in queue
	int size;		//number of element

	Triple* q;		//ptr to Array

	__device__ DeviceQueue();				//do nothing
	__device__ DeviceQueue(Triple*,int); 	//initialize
	__device__ void init(Triple*,int);		//initialize

	__device__ void add(int,int,int);	//add a Triple at the end
	__device__ void pop();				//delete last element
	__device__ Triple* front();			//return last element
	__device__ bool empty();			//return true if empty
	__device__ void print();			//print

	__device__ ~DeviceQueue();			//do nothing
};


////////////////////////////////////////////////////////////////////

__device__ DeviceQueue::DeviceQueue(){}

////////////////////////////////////////////////////////////////////

__device__ DeviceQueue::DeviceQueue(Triple* queue,int nq):q(queue),nQueen(nq),size(nq*nq*3),count(0){}

////////////////////////////////////////////////////////////////////

__device__ void DeviceQueue::init(Triple* queue,int nq){

	q = queue;
	nQueen = nq;
	size = nq*nq*3;
	count = 0;

}

////////////////////////////////////////////////////////////////////

__device__ void DeviceQueue::add(int var, int val, int cs){

	if(count==size){
		ErrorChecking::deviceError("Error::DeviceQueue::add::OUT OF SPACE");
		return;
	}

	q[count].var = var;
	q[count].cs = cs;
	q[count].val = val;

	++count; 

}

////////////////////////////////////////////////////////////////////

__device__ void DeviceQueue::pop(){

	if(count == 0){
		ErrorChecking::deviceError("Error::DeviceQueue::pop::EMPTY QUEUE");
		return;
	}

	--count;

}

////////////////////////////////////////////////////////////////////

__device__ Triple* DeviceQueue::front(){

	if(count == 0){
		ErrorChecking::deviceError("Error::DeviceQueue::front::EMPTY QUEUE");
		return NULL;
	}

	return &q[count-1];

}

////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueue::empty(){

	if(count == 0)return true;
	return false;

}

////////////////////////////////////////////////////////////////////

__device__ void DeviceQueue::print(){

	for(int i = 0; i < count; ++i)
		if(q[i].cs!=5)printf("(%d,%d,%d)\n",q[i].var,q[i].val,q[i].cs);
		else printf("\033[35m(%d,%d,%d)\033[0m\n",q[i].var,q[i].val,q[i].cs);

}

////////////////////////////////////////////////////////////////////

__device__ DeviceQueue::~DeviceQueue(){}

////////////////////////////////////////////////////////////////////
