#include "types.cuh"

// int2 int2
__host__ __device__ int2 operator+(const int2& a, const int2& b) {
	return make_int2(a.x + b.x, a.y + b.y);
}

__host__ __device__ int2 operator-(const int2& a, const int2& b) {
	return make_int2(a.x - b.x, a.y - b.y);
}

__host__ __device__ int2 operator*(const int2& a, const int2& b) {
	return make_int2(a.x * b.x, a.y * b.y);
}

__host__ __device__ int2 operator/(const int2& a, const int2& b) {
	return make_int2(a.x / b.x, a.y / b.y);
}

//int2 int
__host__ __device__ int2 operator+(const int2& a, const int& b) {
	return make_int2(a.x + b, a.y + b);
}

__host__ __device__ int2 operator-(const int2& a, const int& b) {
	return make_int2(a.x - b, a.y - b);
}

__host__ __device__ int2 operator*(const int2& a, const int& b) {
	return make_int2(a.x * b, a.y * b);
}

__host__ __device__ int2 operator/(const int2& a, const int& b) {
	return make_int2(a.x / b, a.y / b);
}

// int3 int3
__host__ __device__ int3 operator+(const int3& a, const int3& b) {
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ int3 operator-(const int3& a, const int3& b) {
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ int3 operator*(const int3& a, const int3& b) {
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ int3 operator/(const int3& a, const int3& b) {
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// float2 float2
__host__ __device__ float2 operator+=(float2& a, const float2& b) {
	a = a + b;
	return a;
}

__host__ __device__ float2 operator+(const float2& a, const float2& b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ float2 operator-(const float2& a, const float2& b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ float2 operator*(const float2& a, const float2& b) {
	return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ float2 operator/(const float2& a, const float2& b) {
	return make_float2(a.x / b.x, a.y / b.y);
}

// float2 float
__host__ __device__ float2 operator+(const float2& a, const float& b) {
	return make_float2(a.x + b, a.y + b);
}

__host__ __device__ float2 operator-(const float2& a, const float& b) {
	return make_float2(a.x - b, a.y - b);
}

__host__ __device__ float2 operator*(const float2& a, const float& b) {
	return make_float2(a.x * b, a.y * b);
}

__host__ __device__ float2 operator*(const float& a, const float2& b) {
	return make_float2(a * b.x, a * b.y);
}

__host__ __device__ float2 operator/(const float2& a, const float& b) {
	return make_float2(a.x / b, a.y / b);
}

// float2 int
__host__ __device__ int2 operator+(const float2& a, const int& b) {
	return make_int2(a.x + b, a.y + b);
}

__host__ __device__ int2 operator-(const float2& a, const int& b) {
	return make_int2(a.x - b, a.y - b);
}

__host__ __device__ int2 operator*(const float2& a, const int& b) {
	return make_int2(a.x * b, a.y * b);
}

__host__ __device__ int2 operator/(const float2& a, const int& b) {
	return make_int2(a.x / b, a.y / b);
}

// float3 float3
__host__ __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

