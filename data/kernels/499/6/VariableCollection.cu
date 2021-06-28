#pragma once
#include "../Variable/Variable.cu"
#include "../TripleQueue/TripleQueue.cu"
#include "../ErrorChecking/ErrorChecking.cu"
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariableCollection{
	int* dMem;							//ptr to deviceMemory
	DeviceVariable* deviceVariableMem;	//vector for variables struct
	int* dMemlastValues;				//last values array
	int nQueen;							//number of variables and also domain size
	HostQueue hostQueue;				//queue

	__host__ HostVariableCollection(int);		//allocate memory with hostMemoryManagemnt
	__host__ ~HostVariableCollection();			//deallocate dMemVariables
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::HostVariableCollection(int nq):
	nQueen(nq),hostQueue(nq){

	ErrorChecking::hostMessage("Warn::HostVariableCollection::constructor::ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariableMem,sizeof(DeviceVariable)*nQueen),"HostVariableCollection::HostVariableCollection::DEVICE VARIABLE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMemlastValues,sizeof(int)*nQueen),"HostVariableCollection::HostVariableCollection::LAST VALUE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMem,sizeof(int)*nQueen*nQueen),"HostVariableCollection::HostVariableCollection::VARIABLE MEM ALLOCATION");
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::~HostVariableCollection(){
	ErrorChecking::hostMessage("Warn::HostVariableCollection::destructor::DELLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableMem),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
	ErrorChecking::hostErrorCheck(cudaFree(dMemlastValues),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
	ErrorChecking::hostErrorCheck(cudaFree(dMem),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariableCollection{

	int fullParallel;				//chose parallel code
	int nQueen;						//number of variables and domain size
	int* lastValues;				//last values array
	int* dMem;	
	DeviceVariable* deviceVariable;	//array for variables
	DeviceQueue deviceQueue;		//triple queue

	__device__ DeviceVariableCollection();											//do nothing
	__device__ DeviceVariableCollection(DeviceVariable*,Triple*, int*,int*,int);	//initialize
	__device__ void init(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ void init2(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ void init3(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ ~DeviceVariableCollection();											//do nothing

	__device__ DeviceVariableCollection& operator=(DeviceVariableCollection&);			//copy

	__device__ bool isGround();			//check if every variable is not failed
	__device__ bool isFailed();			//check if every variable is ground

	__device__ void print();			//print collection

};

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq):
	fullParallel(true),nQueen(nq),deviceVariable(dv),deviceQueue(q,nq),lastValues(lv),dMem(vm){
	
	for(int i = 0; i < nQueen*nQueen; ++i){
		vm[i] = 1;
	}

	for (int i = 0; i < nQueen; ++i){
		deviceVariable[i].init2(&vm[nQueen*i],nQueen);
		lastValues[i]=0;
	}

}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){
	
	dMem = vm;
	fullParallel = true;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

	if(threadIdx.x < nQueen*nQueen){
		vm[threadIdx.x] = 1;
	}

	if(threadIdx.x < nQueen){
		deviceVariable[threadIdx.x].init2(&vm[nQueen*threadIdx.x],nQueen);
		lastValues[threadIdx.x]=0;
	}

}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init2(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){

	fullParallel = true;
	dMem = vm;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

	for (int i = 0; i < nQueen; ++i){
		deviceVariable[i].init2(&vm[nQueen*i],nQueen);
		lastValues[i]=0;
	}

}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init3(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){

	fullParallel = true;
	dMem = vm;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::~DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isGround(){
	for(int i = 0; i < nQueen; ++i)
		if(deviceVariable[i].ground==-1)return false;

	return true;
}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isFailed(){
	for(int i = 0; i < nQueen; ++i)
		if(deviceVariable[i].failed == 1)return true;

	return false;
}

///////////////////////////////////////////////////////////////////////

__device__ void externCopy(DeviceVariableCollection& to,DeviceVariableCollection& other){

	__shared__ int nQueen; 
	__shared__ int next1; 
	__shared__ int next2; 
	__shared__ int next3;

	nQueen = to.nQueen;
	
	next1 = ((((int(3*nQueen*nQueen/32)+1)*32)-3*nQueen*nQueen)+3*nQueen*nQueen);
	next2 = ((((int((next1+nQueen*nQueen)/32)+1)*32)-(next1+nQueen*nQueen))+(next1+nQueen*nQueen));
	next3 = ((((int((next2+nQueen)/32)+1)*32)-(next2+nQueen))+(next2+nQueen));

	if(threadIdx.x < 3*nQueen*nQueen)
		to.deviceQueue.q[threadIdx.x] = other.deviceQueue.q[threadIdx.x];

	if(threadIdx.x >=  next1 && threadIdx.x < next1 + nQueen*nQueen)
		to.dMem[threadIdx.x - next1] = other.dMem[threadIdx.x - next1];

	if(threadIdx.x >= next2 && threadIdx.x < next2 + nQueen)
		to.lastValues[threadIdx.x - next2] = other.lastValues[threadIdx.x- next2];

	if(threadIdx.x >= next3 && threadIdx.x < next3 + nQueen){
		to.deviceVariable[threadIdx.x - next3].ground = other.deviceVariable[threadIdx.x - next3].ground;
		to.deviceVariable[threadIdx.x - next3].failed = other.deviceVariable[threadIdx.x - next3].failed;
		to.deviceVariable[threadIdx.x - next3].changed = other.deviceVariable[threadIdx.x - next3].changed;
	}

	if(threadIdx.x == 1023)
		to.deviceQueue.count = other.deviceQueue.count;

}

__device__ DeviceVariableCollection& DeviceVariableCollection::operator=(DeviceVariableCollection& other){

/*	__shared__ int next1; 
	__shared__ int next2; 
	__shared__ int next3;

	next1 = ((((int(3*nQueen*nQueen/32)+1)*32)-3*nQueen*nQueen)+3*nQueen*nQueen);
	next2 = ((((int((next1+nQueen*nQueen)/32)+1)*32)-(next1+nQueen*nQueen))+(next1+nQueen*nQueen));
	next3 = ((((int((next2+nQueen)/32)+1)*32)-(next2+nQueen))+(next2+nQueen));

	if(threadIdx.x < 3*nQueen*nQueen)
		this->deviceQueue.q[threadIdx.x] = other.deviceQueue.q[threadIdx.x];

	if(threadIdx.x >=  next1 && threadIdx.x < next1 + nQueen*nQueen)
		this->dMem[threadIdx.x - next1] = other.dMem[threadIdx.x - next1];

	if(threadIdx.x >= next2 && threadIdx.x < next2 + nQueen)
		this->lastValues[threadIdx.x - next2] = other.lastValues[threadIdx.x- next2];

	if(threadIdx.x >= next3 && threadIdx.x < next3 + nQueen){
		this->deviceVariable[threadIdx.x - next3].ground = other.deviceVariable[threadIdx.x - next3].ground;
		this->deviceVariable[threadIdx.x - next3].failed = other.deviceVariable[threadIdx.x - next3].failed;
		this->deviceVariable[threadIdx.x - next3].changed = other.deviceVariable[threadIdx.x - next3].changed;
	}

	if(threadIdx.x == 1023)
		this->deviceQueue.count = other.deviceQueue.count;*/

	if(threadIdx.x < 3*nQueen*nQueen)
		this->deviceQueue.q[threadIdx.x] = other.deviceQueue.q[threadIdx.x];

	if(threadIdx.x < nQueen*nQueen)
		this->dMem[threadIdx.x] = other.dMem[threadIdx.x];

	if(threadIdx.x < nQueen)
		this->lastValues[threadIdx.x] = other.lastValues[threadIdx.x];

	if(threadIdx.x < nQueen){
		this->deviceVariable[threadIdx.x].ground = other.deviceVariable[threadIdx.x].ground;
		this->deviceVariable[threadIdx.x].failed = other.deviceVariable[threadIdx.x].failed;
		this->deviceVariable[threadIdx.x].changed = other.deviceVariable[threadIdx.x].changed;
	}

	if(threadIdx.x == 1023)
		this->deviceQueue.count = other.deviceQueue.count;

	return *this;
}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::print(){
	for (int i = 0; i < nQueen; ++i){
		printf("[%d] ::: ",lastValues[i]);
		deviceVariable[i].print();
	}
	deviceQueue.print();
	printf("\n");
}

///////////////////////////////////////////////////////////////////////
