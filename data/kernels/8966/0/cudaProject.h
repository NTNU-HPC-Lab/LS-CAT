#ifndef __CUDAPROJECT_H
#define __CUDAPROJECT_H

class __declspec(dllexport) cudaProject
{
public:
	cudaProject() {}
	~cudaProject() {}
	void mathVectors(float* c, float* a, float* b, int n, int oper);
};

#endif // __CUDAPROJECT_H
