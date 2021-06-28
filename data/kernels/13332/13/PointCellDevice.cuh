/*
 * PointCellDevice.cuh
 *
 *  Created on: 07.10.2016
 *      Author: basti
 */

#ifndef POINTCELLDEVICE_CUH_
#define POINTCELLDEVICE_CUH_

#include "data.cuh"
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <cuda.h>


class PointCellDevice {
public:
	PointCellDevice();
	virtual ~PointCellDevice();

	/*
	 * data = [stateVector | stateCopy | F | P | H | R | K | I | Q | S | tmp | tmp2]
	 */
	float data[260];

	float subInvtl;

	__host__ __device__ void predict();
	__host__ __device__ void update(float* newState);
	__host__ __device__ int getID();
	__host__ __device__ void setID(int id);
	__host__ __device__ float getX();
	__host__ __device__ float getY();
	__host__ __device__ float getTheta();
	__host__ __device__ float getVelocity();
	__host__ __device__ float getPhi();

	__host__ __device__ void setX(float x);
	__host__ __device__ void setY(float y);
	__host__ __device__ void setTheta(float theta);
	__host__ __device__ void setVelocity(float velocity);
	__host__ __device__ void setPhi(float phi);
	__host__ __device__ void initializeMemory();

	__host__ __device__ void computeF();
	__host__ __device__ void computeCovarianceF();
	__host__ __device__ void writeP(int row, int col, float value);
	__host__ __device__ void writeQ(int row, int col, float value);
	__host__ __device__ void writeR(int row, int col, float value);
	__host__ __device__ void writeH(int row, int col, float value);
	__host__ __device__ void writeK(int row, int col, float value);
	__host__ __device__ void writeI(int row, int col, float value);
	__host__ __device__ void writeF(int row, int col, float value);
	__host__ __device__ void writeS(int row, int col, float value);
	__host__ __device__ void writeTmp(int row, int col, float value);
	__host__ __device__ void writeTmp2(int row, int col, float value);

	__host__ __device__ float getP(int row, int col);
	__host__ __device__ float getQ(int row, int col);
	__host__ __device__ float getR(int row, int col);
	__host__ __device__ float getH(int row, int col);
	__host__ __device__ float getK(int row, int col);
	__host__ __device__ float getI(int row, int col);
	__host__ __device__ float getF(int row, int col);
	__host__ __device__ float getS(int row, int col);
	__host__ __device__ float getTmp(int row, int col);
	__host__ __device__ float getTmp2(int row, int col);

	__host__ __device__ void invertS();

private:


	__host__ __device__ void reducedRowEcholon(float* toInvert);
	__host__ __device__ void reorder(float* toInvert, int* order);
	__host__ __device__ void divideRow(float* toInvert, int row, float divisor);
	__host__ __device__ void rowOperation(float* toInvert, int row, int addRow, float scale);
	__host__ __device__ unsigned getLeadingZeros(unsigned row, float* toInvert) const;
	__host__ __device__ void getSubMatrix(float* toInvert, unsigned startRow,unsigned endRow,unsigned startColumn,unsigned endColumn, int* newOrder = NULL);

	int ID;
};

#endif /* POINTCELLDEVICE_CUH_ */
