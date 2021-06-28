#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

__host__ __device__ float3 operator*(float a, const float3 &b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ float3 operator*(const float3 &b, float a) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ float3 operator/(const float3 &b, float a) {
	return make_float3(b.x / a, b.y / a, b.z / a);
}

__host__ __device__ float3 operator/=(float3 a, float b) {
	float t = 1 / b;
	return make_float3(a.x *= t, a.y *= t, a.z *= t);
}

__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator+=(float3 &a, const float3 &b) {
	return make_float3(a.x += b.x, a.y += b.y, a.z += b.z);
}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float dot(const float3 &a, const float3 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float squared_length(const float3 &a) {
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

__host__ __device__ float length(const float3 &a) {
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ float3 unit_vector(const float3 & a) {
	return a / length(a);
}
#endif