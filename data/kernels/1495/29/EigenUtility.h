#pragma once
#include <iostream>
#include <cmath>

using namespace std;

class EigenUtility
{
public:
	EigenUtility();
	~EigenUtility();

	// 外部呼叫
	void SetAverageValue(float);				// 這邊是去設定平均值
	void SolveByEigen(float*, float*, int);		// 這邊是去解 Eigen
	float* GetFunctionArray(int, int);			// 先算出 Size

	float *params;								// 參數式多少

private:
	//////////////////////////////////////////////////////////////////////////
	// Function 一些參數
	//////////////////////////////////////////////////////////////////////////
	float avg;						// Y 的平均
	int NumPolynomial = -1;			// 有幾次項
	
};

