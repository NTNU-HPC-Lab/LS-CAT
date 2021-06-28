#pragma once
#include "../VariableCollection/VariableCollection.cu"
#include "../ErrorChecking/ErrorChecking.cu"

struct DeviceQueenPropagation{

	//////////////////////////////////////SINGLE THREAD//////////////////////////////////////

	__device__ int static inline nextAssign(DeviceVariableCollection&,int);		//assign next value not already tried
																	//returns assigned value

	__device__ int static inline allDifferent(DeviceVariableCollection&,int,int,int);		//propagate for all different constraint code 3
	__device__ int static inline diagDifferent(DeviceVariableCollection&,int,int,int);	//propagate for diag constraint code 4

	__device__ int static inline sequentialForwardChecking(DeviceVariableCollection&,int,int);	//csp forward propagation code 5
	__device__ int static inline sequentialBacktracking(DeviceVariableCollection&);		//csp undo forward propagation

	//////////////////////////////////////MULTI THREAD//////////////////////////////////////

	__device__ int static inline parallelForwardChecking(DeviceVariableCollection&,int,int);
	__device__ int static inline parallelForwardChecking(DeviceVariableCollection&,int,int,cudaStream_t&);
	__device__ int static inline parallelBacktracking(DeviceVariableCollection&);		//csp undo forward propagation
};

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::nextAssign(DeviceVariableCollection& vc, int var){

/*	if(var < 0 || var >= vc.nQueen){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::nextAssign::VAR OUT OF BOUND");
		return -1;
	}

	if(vc.lastValues[var] >= vc.nQueen){
		if(threadIdx.x == 0)ErrorChecking::deviceMessage("Warn::DeviceQueenPropagation::nextAssign::VALUE OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].failed == 1){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::nextAssign::VAR ALREADY FAILED");
		return -1;
	}*/

	__shared__ int nextAss;
	nextAss = -1;

	__syncthreads();

	if(threadIdx.x == 0){
		int next;
		for(next = vc.lastValues[var];next<vc.nQueen;++next){
			if(vc.deviceVariable[var].domain[next]==1){
				vc.lastValues[var]=next+1;
				nextAss = next;
				break;
			}
		}
	}

	__syncthreads();

	if(nextAss != -1){

		if(threadIdx.x < vc.nQueen && threadIdx.x != nextAss){
			--vc.deviceVariable[var].domain[threadIdx.x];
		}

		if(threadIdx.x == 0){
			vc.deviceVariable[var].ground = nextAss;
		}

	}

	//if(threadIdx.x == 0)ErrorChecking::deviceMessage("Warn::DeviceQueenPropagation::nextAssign::NEXTVALUE NOT FOUND");


	return nextAss;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::allDifferent(DeviceVariableCollection& vc, int var, int val, int delta){

/*	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::allDifferent::OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::QueenPropagation::allDifferent::VARIABLE NOT GROUND");
		return -1;
	}*/
	
	for(int i = 0; i < vc.nQueen; ++i)
		if(i != var){
			vc.deviceVariable[i].addTo(val,delta);

		}
	
	if(delta < 0)vc.deviceQueue.add(var,val,3);

	return 0;	

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::diagDifferent(DeviceVariableCollection& vc, int var, int val, int delta){

/*	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::diagDifferent::OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::diagDifferent::VARIABLE NOT GROUND");
		return -1;
	}*/

	int i=var+1,j=val+1;
	while(i<vc.nQueen && j<vc.nQueen){
		vc.deviceVariable[i].addTo(j,delta);
		++i;++j;
	}

	i=var-1,j=val-1;
	while(i>=0 && j>=0){
		vc.deviceVariable[i].addTo(j,delta);
		--i;--j;
	}

	i=var-1,j=val+1;
	while(i>=0 && j<vc.nQueen){
		vc.deviceVariable[i].addTo(j,delta);
		--i;++j;
	}

	i=var+1,j=val-1;
	while(i<vc.nQueen && j>=0){
		vc.deviceVariable[i].addTo(j,delta);
		++i;--j;
	}

	if(delta < 0)vc.deviceQueue.add(var,val,4);
	return 0;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::sequentialForwardChecking(DeviceVariableCollection& vc, int var, int val){

/*	if(var < 0 || var > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::sequentialForwardChecking:: VAR OUT OF BOUND");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::sequentialForwardChecking:: VAL OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::sequentialForwardChecking::VARIABLE NOT GROUND");
		return -1;
	}*/

	allDifferent(vc,var,val,-1);
	diagDifferent(vc,var,val,-1);

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.deviceVariable[i].changed==1){
				if(vc.deviceVariable[i].ground>=0){
					allDifferent(vc,i,vc.deviceVariable[i].ground,-1);
					diagDifferent(vc,i,vc.deviceVariable[i].ground,-1);
					ch = true;
				}
				vc.deviceVariable[i].changed=-1;
			}
		}
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	if (vc.isFailed()) return -1;

	return 0;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::sequentialBacktracking(DeviceVariableCollection& vc){

/*	if(vc.deviceQueue.front()->cs!=5){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::sequentialBacktracking::ERROR IN QUEUE");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::sequentialBacktracking::EMPTY QUEUE");
		return -1;		
	}*/

	int t1=vc.deviceQueue.front()->var;
	int t2=vc.deviceQueue.front()->val;

	for(int i = t1+1; i < vc.nQueen; ++i)vc.lastValues[i]=0;

	vc.deviceQueue.pop();
	while(vc.deviceQueue.front()->cs!=5){
		switch(vc.deviceQueue.front()->cs){
			case 3:{
				allDifferent(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,+1);	
			}break;
			case 4:{
				diagDifferent(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,+1);	
			}break;
		}
		vc.deviceQueue.pop();

		if(vc.deviceQueue.empty())break;
	}

	vc.deviceVariable[t1].undoAssign(t2);
	return 0;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::parallelForwardChecking(DeviceVariableCollection& vc, int var, int val){

/*	if(var < 0 || var > vc.nQueen){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAR OUT OF BOUND");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAL OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VARIABLE NOT GROUND");
		return -1;
	}

	__syncthreads();*/

	{
		int columnIndex = threadIdx.x % vc.nQueen;
		int rowIndex = int(threadIdx.x/vc.nQueen);

		__shared__ bool ch;

		if(threadIdx.x < vc.nQueen*vc.nQueen){
			if(rowIndex != var && val == columnIndex){

				int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
				if(old == 1){
					vc.deviceVariable[rowIndex].changed = 1;
				}

			}
			
			if(rowIndex != var && columnIndex == rowIndex && columnIndex+val-var < vc.nQueen && columnIndex+val-var >= 0){

				int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+val-var],-1);
				if(old == 1){
					vc.deviceVariable[rowIndex].changed = 1;
				}

			}
			
			if(rowIndex != var && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-val)+var < vc.nQueen && columnIndex-(vc.nQueen-val)+var >= 0){

				int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-val)+var],-1);
				if(old == 1){
					vc.deviceVariable[rowIndex].changed = 1;
				}

			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			int old = atomicAdd(&vc.deviceQueue.count,1);
			vc.deviceQueue.q[old].var = var;
			vc.deviceQueue.q[old].val = val;
			vc.deviceQueue.q[old].cs = 6;

		}

		if(threadIdx.x >= vc.nQueen && threadIdx.x < vc.nQueen*2)
			vc.deviceVariable[threadIdx.x-vc.nQueen].checkFailed();

		if(threadIdx.x >= vc.nQueen*2 && threadIdx.x < vc.nQueen*3)
			vc.deviceVariable[threadIdx.x-vc.nQueen*2].checkGround();

		do{
			
			__syncthreads();
			
			ch=false;
			
			for(int i = var+1; i < vc.nQueen; ++i){


				if(vc.deviceVariable[i].changed == 1){

					if(vc.deviceVariable[i].ground>=0){

						__syncthreads();

						if(threadIdx.x < vc.nQueen*vc.nQueen){
							if(rowIndex != i && vc.deviceVariable[i].ground == columnIndex){

								int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
								if(old == 1){
									vc.deviceVariable[rowIndex].changed = 1;
								}

							}
							
							if(rowIndex != i && columnIndex == rowIndex && columnIndex+vc.deviceVariable[i].ground-i < vc.nQueen && columnIndex+vc.deviceVariable[i].ground-i >= 0){

								int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+vc.deviceVariable[i].ground-i],-1);
								if(old == 1){
									vc.deviceVariable[rowIndex].changed = 1;
								}

							}
							
							if(rowIndex != i && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i < vc.nQueen && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i >= 0){

								int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i],-1);
								if(old == 1){
									vc.deviceVariable[rowIndex].changed = 1;
								}

							}
						}

						__syncthreads();

						if(threadIdx.x == 0){
							int old = atomicAdd(&vc.deviceQueue.count,1);
							vc.deviceQueue.q[old].var = i;
							vc.deviceQueue.q[old].val = vc.deviceVariable[i].ground;
							vc.deviceQueue.q[old].cs = 6;

						}

						if(threadIdx.x >= vc.nQueen && threadIdx.x < vc.nQueen*2)
							vc.deviceVariable[threadIdx.x-vc.nQueen].checkFailed();

						if(threadIdx.x >= vc.nQueen*2 && threadIdx.x < vc.nQueen*3)
							vc.deviceVariable[threadIdx.x-vc.nQueen*2].checkGround();

						ch = true;
					}

					__syncthreads();
					vc.deviceVariable[i].changed=-1;
				}
			}

			if(vc.isFailed())ch = false;

		}while(ch);

	}

	__syncthreads();

	if(vc.isFailed()){
		__syncthreads();
		if(threadIdx.x < vc.nQueen)vc.deviceVariable[threadIdx.x].changed = -1;
		if(threadIdx.x == 0)vc.deviceQueue.add(var,val,5);
		__syncthreads();
		return 1;
	}

	__syncthreads();
	if(threadIdx.x == 0)vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::parallelBacktracking(DeviceVariableCollection& vc){

/*	if(vc.deviceQueue.front()->cs!=5){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelBacktracking::ERROR IN QUEUE");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		if(threadIdx.x == 0)ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelBacktracking::EMPTY QUEUE");
		return -1;		
	}*/

	__shared__ int t1;
	__shared__ int t2;

	t1 = vc.deviceQueue.front()->var;
	t2 = vc.deviceQueue.front()->val;

	__syncthreads();

	if(threadIdx.x == 0){
		vc.deviceQueue.pop();
	}

	if(threadIdx.x >= t1+1 && threadIdx.x < vc.nQueen){
		vc.lastValues[threadIdx.x]=0;
	}

	__syncthreads();

	while(vc.deviceQueue.front()->cs!=5 && !vc.deviceQueue.empty()){

		int col = threadIdx.x % vc.nQueen;
		int row = int(threadIdx.x/vc.nQueen);

		int var = vc.deviceQueue.front()->var;
		int val = vc.deviceQueue.front()->val;

		if(threadIdx.x < vc.nQueen*vc.nQueen){
			if(row != var && val == col){
				atomicAdd(&vc.deviceVariable[row].domain[col],1);
			}
			
			if(row != var && col == row && col+val-var < vc.nQueen && col+val-var >= 0){
				atomicAdd(&vc.deviceVariable[row].domain[col+val-var],1);
			}
			
			if(row != var && vc.nQueen-col == row && col-(vc.nQueen-val)+var < vc.nQueen && col-(vc.nQueen-val)+var >= 0){
				atomicAdd(&vc.deviceVariable[row].domain[col-(vc.nQueen-val)+var],1);
			}
		}

		__syncthreads();

		if(threadIdx.x == 0)vc.deviceQueue.pop();

		__syncthreads();
	}

	if(threadIdx.x < vc.nQueen && threadIdx.x != t2){
		++vc.deviceVariable[t1].domain[threadIdx.x];
	}

	__syncthreads();

	if(threadIdx.x < vc.nQueen) vc.deviceVariable[threadIdx.x].checkFailed();
	if(threadIdx.x >= vc.nQueen && threadIdx.x <2*vc.nQueen) vc.deviceVariable[threadIdx.x-vc.nQueen].checkGround();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
