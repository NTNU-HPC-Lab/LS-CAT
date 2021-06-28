#include <iostream>
using namespace std;

struct vec2D {
	double x;
	double y;

	__host__ __device__ vec2D(){

	}
	__host__ __device__ vec2D(double x, double y){
		this->x = x;
		this->y = y;
	}

	__host__ __device__ double length(){
		return x*x + y*y;
	}

	__host__ __device__ friend vec2D operator*(vec2D a, vec2D b){
		vec2D c = vec2D(a.x * b.x, a.y * b.y);
		return c;
	}

	__host__ __device__ friend vec2D operator-(vec2D a, vec2D b){
		vec2D c = vec2D(a.x - b.x, a.y - b.y);
		return c;
	}

	__host__ __device__ friend vec2D operator+(vec2D a, vec2D b){
		vec2D c = vec2D(a.x + b.x, a.y + b.y);
		return c;
	}

	__host__ __device__ friend vec2D operator*(double a, vec2D v){
		vec2D c = vec2D(v.x * a, v.y * a);
		return c;
	}


	friend ostream& operator<<(ostream& os, vec2D a){
		os<<a.x<<" "<<a.y<<endl;
		return os;
	};
	

};