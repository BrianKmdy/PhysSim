#include "Operations.cuh"

__host__ __device__ float2 direction(float2 a, float2 b)
{
	return b - a;
}

// XXX/bmoody Review the order
//            Can probably make this more efficient by skippint the sqrt
__host__ __device__ float distance(float2 a, float2 b)
{
	return sqrtf(powf(b.x - a.x, 2.0) + powf(b.y - a.y, 2.0));
}

__host__ __device__ float magnitude(float2 a)
{
	return sqrtf(powf(a.x, 2.0) + powf(a.y, 2.0));
}

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
__host__ __device__ float2 operator+(const float2& a, const int& b) {
	return make_float2(a.x + b, a.y + b);
}

__host__ __device__ float2 operator-(const float2& a, const int& b) {
	return make_float2(a.x - b, a.y - b);
}

__host__ __device__ float2 operator*(const float2& a, const int& b) {
	return make_float2(a.x * b, a.y * b);
}

__host__ __device__ float2 operator/(const float2& a, const int& b) {
	return make_float2(a.x / b, a.y / b);
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

// Absolute value
__host__ __device__ int2 abs(const int2& a)
{
	return make_int2(abs(a.x), abs(a.y));
}