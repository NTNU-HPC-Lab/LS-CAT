#include "PointCellDevice.cuh"

/*
 * data* = [stateVector | stateCopy | F | P | H | R | K | I | Q | S | tmp | tmp2]
 */
PointCellDevice::PointCellDevice()
{
	initializeMemory();
	subInvtl = 0.5;
}

PointCellDevice::~PointCellDevice()
{
}
/*
 * set initial values to all matrices and vectors
 */
__host__ __device__ void PointCellDevice::initializeMemory()
{
	//initialize data to 0
	for(int i=0; i<260; i++)
	{
		data[i] = 0;
	}

	for(int i=0; i<5; i++)
	{
		//P
		data[35 + i*5 + i] = 1000;
		//F
		data[10 + i*5 + i] = 1;
		//I
		data[135 + i*5 + i] = 1;
		//H
		data[60 + i*5 + i] = 1;
	}
	//F(2,4)
	data[10 + 2*5 +4] = TIMESTAMP;

	//Q
	data[160] = 0.000006f;
	data[160 + 1*5 + 1] = 0.000006f;
	data[160 + 2*5 + 2] = 0.0004f;
	data[160 + 3*5 + 3] = 0.03097f;
	data[160 + 4*5 + 4] = 0.0004f;

	//R
	data[85] = 0.36f;
	data[85 + 1*5 + 1] = 0.36f;
	data[85 + 2*5 + 2] = 0.5f;
	data[85 + 3*5 + 3] = 0.1f;
	data[85 + 4*5 + 4] = 0.1f;
}
/*
 * perfrom kalman filter predict step
 */
__host__ __device__ void PointCellDevice::predict()
{
	//store copy of stateVector
	for(int i=0; i<5; i++)
	{
		data[i+5] = data[i];
	}
	//estimate new state
	computeF();
	//compute new state covariance
	computeCovarianceF();

	float tmp = 0;
	// Tmp = F*P
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getF(i,k)*getP(k,j);
			}
			writeTmp(i,j, tmp);
		}
	}

	//P = Tmp*F_t
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getTmp(i,k)*getF(j,k);
			}
			writeP(i,j, tmp);
		}
	}

	//P = P+Q
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = getP(i,j) + getQ(i,j);
			writeP(i,j, tmp);
		}
	}
}
/*
 * estimates new state
 */
__host__ __device__ void PointCellDevice::computeF()
{
	float x = getX();
	float y = getY();
	float theta = getTheta();
	float velocity = getVelocity();
	float phi = getPhi();

	float predictedX, predictedY, predictedTheta, predictedVel,predictedPhi;

	if(phi > 0.0001)
	{
		predictedX = (velocity/phi) * (sinf(phi*TIMESTAMP + theta) - sinf(theta)) + x;
		predictedY = (velocity/phi) * (-cosf(phi*TIMESTAMP + theta) + cosf(theta)) + y;
		predictedTheta = phi*TIMESTAMP + theta;
		predictedVel = velocity;
		predictedPhi = phi;
	}
	else
	{
		predictedX = x + velocity * TIMESTAMP * cosf(theta);
		predictedY = y + velocity * TIMESTAMP * sinf(theta);
		predictedTheta = theta;
		predictedVel = velocity;
		predictedPhi = 0.00001;
	}

	setX(predictedX);
	setY(predictedY);
	setTheta(predictedTheta);
	setVelocity(predictedVel);
	setPhi(predictedPhi);
}
/*
 * computes new state covariance
 */
__host__ __device__ void PointCellDevice::computeCovarianceF()
{
	float theta = getTheta();
	float velocity = getVelocity();
	float phi = getPhi();

	float f12, f13, f14, f22, f23, f24;

	f12 = (velocity/phi) * (-cosf(theta) + cosf(TIMESTAMP*phi + theta));
	f13 = (1/phi) * (sinf(phi*TIMESTAMP + theta) - sinf(theta));
	f14 = (((TIMESTAMP*velocity)/phi) * cosf(TIMESTAMP*phi + theta)) - ((velocity/(phi*phi)) * (sinf(phi*TIMESTAMP + theta) - sinf(theta)));

	f22 = (velocity/phi) * (sinf(phi*TIMESTAMP + theta) - sinf(theta));
	f23 = (1/phi) * (-cosf(phi*TIMESTAMP + theta) + cosf(theta));
	f24 = (((TIMESTAMP*velocity)/phi) * sinf(TIMESTAMP*phi + theta)) - ((velocity/(phi*phi)) * (-cosf(phi*TIMESTAMP + theta) + cosf(theta)));

	writeF(0,2,f12);
	writeF(0,3,f13);
	writeF(0,4,f14);
	writeF(1,2,f22);
	writeF(1,3,f23);
	writeF(1,4,f24);
}
/*
 * perfomrs kalman filter update step with given new state
 */
__host__ __device__ void PointCellDevice::update(float* newState)
{
	float velocity, phi;
	float xNew = newState[0];
	float yNew = newState[1];
	float thetaNew = newState[2];

	float x = data[5];
	float y = data[6];
	float theta = data[7];
	//first compute yawrate and velocity based in new and old position
	velocity = sqrtf((xNew - x) * (xNew - x) + (yNew - y)*(yNew - y)) / TIMESTAMP;
	phi = (thetaNew-theta) / TIMESTAMP;

	setVelocity(velocity);
	setPhi(phi);
	float tmp = 0;

	//tmp = H*P
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getH(i,k)*getP(k,j);
			}
			writeTmp(i,j, tmp);
		}
	}

	//S = tmp*H_t
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getTmp(i,k)*getH(j,k);
			}
			writeS(i,j, tmp);
		}
	}

	//S = S+R
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = getS(i,j) + getR(i,j);
			writeS(i,j, tmp);
		}
	}

	//tmp = P*H_t
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getP(i,k)*getH(j,k);
			}
			writeTmp(i,j, tmp);
		}
	}

	invertS();

	//K = tmp*S_i
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getTmp(i,k)*getS(k,j);
			}
			writeK(i,j, tmp);
		}
	}

	//tmp = K*(newState-stateVector)
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<1; j++)
		{
			tmp = 0;
			tmp += getK(i,0)*(xNew-getX());
			tmp += getK(i,1)*(yNew-getY());
			tmp += getK(i,2)*(thetaNew-getTheta());
			tmp += getK(i,3)*(velocity-getVelocity());
			tmp += getK(i,4)*(phi-getPhi());
			writeTmp(i,j, tmp);
		}
	}

	//stateVector = stateVector + tmp
	setX(getX() + getTmp(0,0));
	setY(getY() + getTmp(1,0));
	setTheta(getTheta() + getTmp(2,0));
	setVelocity(getVelocity() + getTmp(3,0));
	setPhi(getPhi() + getTmp(4,0));

	//tmp = K*H
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getK(i,k)*getH(k,j);
			}
			writeTmp(i,j, tmp);
		}
	}

	//tmp = I - tmp
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = getI(i,j) - getTmp(i,j);
			writeTmp(i,j, tmp);
		}
	}

	//tmp2 = tmp*P
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			tmp = 0;
			for(int k=0; k<5; k++)
			{
				tmp += getTmp(i,k)*getP(k,j);
			}
			writeTmp2(i,j, tmp);
		}
	}

	for(int i=0; i<5;i++)
	{
		for(int j=0; j<5; j++)
		{
			writeP(i,j, getTmp2(i,j));
		}
	}

}
/*
 * performs the inversion of S
 */
__host__ __device__ void PointCellDevice::invertS()
{
	//Concatenate IdentityMatrix to the right of S
	float toInvert[50];
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<5; j++)
		{
			toInvert[i*10 +j] = getS(i,j);
		}
	}
	for(int i=0; i<5; i++)
	{
		for(int j=5; j<10; j++)
		{
			if(j-i == 5)
			{
				toInvert[i*10 +j] = 1;
			}
			else
			{
				toInvert[i*10 +j] = 0;
			}
		}
	}
	reducedRowEcholon(toInvert);

}

__host__ __device__ void PointCellDevice::reducedRowEcholon(float* toInvert)
{
    float const ZERO = static_cast<float>( 0 );
    int order[5];
    int rows = 5;
    int columns = 10;
    for(int i=0; i<5; i++)
    {
    	order[i] = i;
    }
    // For each row...
    for ( unsigned rowIndex = 0; rowIndex < rows; ++rowIndex )
    {
      // Reorder the rows.
      reorder(toInvert, order);

      unsigned row = order[ rowIndex ];

      // Divide row down so first term is 1.
      unsigned column = getLeadingZeros( row , toInvert);
      float divisor = toInvert[(row * columns) + column];
      if ( ZERO != divisor )
      {
        divideRow(toInvert, row, divisor );

        // Subtract this row from all subsequent rows.
        for ( unsigned subRowIndex = ( rowIndex + 1 ); subRowIndex < rows; ++subRowIndex )
        {
          unsigned subRow = order[ subRowIndex ];
          if ( ZERO != toInvert[(subRow * columns) + column] )
            rowOperation
            (
              toInvert,
              subRow,
              row,
              -toInvert[(subRow * columns) + column]
            );
        }
      }

    }

    // Back substitute all lower rows.
    for ( unsigned rowIndex = ( rows - 1 ); rowIndex > 0; --rowIndex )
    {
      unsigned row = order[ rowIndex ];
      unsigned column = getLeadingZeros( row ,toInvert);
      for ( unsigned subRowIndex = 0; subRowIndex < rowIndex; ++subRowIndex )
      {
        unsigned subRow = order[ subRowIndex ];
        rowOperation
        (
          toInvert,
          subRow,
          row,
          -toInvert[(subRow * columns) + column]
        );
      }
    }
    getSubMatrix(toInvert,0, 4, 5, 9, order);
}
__host__ __device__ void PointCellDevice::reorder(float* toInvert, int* order)
{
    unsigned zeros[5];
    int rows = 5;
    for ( unsigned row = 0; row < rows; ++row )
    {
      order[ row ] = row;
      zeros[ row ] = getLeadingZeros(row, toInvert);
    }

    for ( unsigned row = 0; row < (rows-1); ++row )
    {
      unsigned swapRow = row;
      for ( unsigned subRow = row + 1; subRow < rows; ++subRow )
      {
        if ( zeros[ order[ subRow ] ] < zeros[ order[ swapRow ] ] )
          swapRow = subRow;
      }

      unsigned hold    = order[ row ];
      order[ row ]     = order[ swapRow ];
      order[ swapRow ] = hold;
    }
}
__host__ __device__ void PointCellDevice::divideRow(float* toInvert, int row, float divisor)
{
    for ( unsigned column = 0; column < 10; ++column )
    {
      toInvert[ (row * 10) + column] /= divisor;
    }
}
__host__ __device__ void PointCellDevice::rowOperation(float* toInvert, int row, int addRow, float scale)
{
	int columns = 10;
    for ( unsigned column = 0; column < columns; ++column )
    {
      toInvert[ (row * columns) + column] += toInvert[ (addRow * columns) + column] * scale;
    }
}

__host__ __device__ unsigned PointCellDevice::getLeadingZeros(unsigned row, float* toInvert) const
{
	  float const ZERO = static_cast< float >( 0 );
	  unsigned column = 0;
	  while ( ZERO == toInvert[ (row * 10) + column] )
	  {
	    ++column;
	  }
	  return column;
}

__host__ __device__ void PointCellDevice::getSubMatrix(float* toInvert, unsigned startRow,unsigned endRow,unsigned startColumn,unsigned endColumn, int* newOrder)
{
	int columns = 10;
    for ( unsigned row = startRow; row <= endRow; ++row )
    {
      unsigned subRow;
      if ( NULL == newOrder )
        subRow = row;
      else
        subRow = newOrder[ row ];

      for ( unsigned column = startColumn; column <= endColumn; ++column )
      {
    	 writeS((row - startRow),(column - startColumn), toInvert[ (subRow * columns) + column]);
      }
    }

}
__host__ __device__ int PointCellDevice::getID()
{
	return ID;
}
__host__ __device__ void PointCellDevice::setID(int id)
{
	ID = id;
}
__host__ __device__ float PointCellDevice::getX()
{
	return data[0];
}
__host__ __device__ float PointCellDevice::getY()
{
	return data[1];
}
__host__ __device__ float PointCellDevice::getTheta()
{
	return data[2];
}
__host__ __device__ float PointCellDevice::getVelocity()
{
	return data[3];
}
__host__ __device__ float PointCellDevice::getPhi()
{
	return data[4];
}

__host__ __device__ void PointCellDevice::setX(float x)
{
	data[0] = x;
}
__host__ __device__ void PointCellDevice::setY(float y)
{
	data[1] = y;
}
__host__ __device__ void PointCellDevice::setTheta(float theta)
{
	data[2] = theta;
}
__host__ __device__ void PointCellDevice::setVelocity(float velocity)
{
	data[3] = velocity;
}
__host__ __device__ void PointCellDevice::setPhi(float phi)
{
	data[4] = phi;
}

__host__ __device__ void PointCellDevice::writeP(int row, int col, float value)
{
	data[35 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeF(int row, int col, float value)
{
	data[10 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeH(int row, int col, float value)
{
	data[60 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeR(int row, int col, float value)
{
	data[85 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeK(int row, int col, float value)
{
	data[110 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeI(int row, int col, float value)
{
	data[135 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeQ(int row, int col, float value)
{
	data[160 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeS(int row, int col, float value)
{
	data[185 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeTmp(int row, int col, float value)
{
	data[210 + row*5 + col] = value;
}

__host__ __device__ void PointCellDevice::writeTmp2(int row, int col, float value)
{
	data[235 + row*5 + col] = value;
}

__host__ __device__ float PointCellDevice::getP(int row, int col)
{
	return data[35 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getF(int row, int col)
{
	return data[10 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getH(int row, int col)
{
	return data[60 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getR(int row, int col)
{
	return data[85 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getK(int row, int col)
{
	return data[110 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getI(int row, int col)
{
	return data[135 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getQ(int row, int col)
{
	return data[160 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getS(int row, int col)
{
	return data[185 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getTmp(int row, int col)
{
	return data[210 + row*5 + col];
}

__host__ __device__ float PointCellDevice::getTmp2(int row, int col)
{
	return data[235 + row*5 + col];
}
