#pragma once

#include "cuda_runtime.h"

__device__ float3 operator+(const float3& a, const float3& b);
__device__ float3 operator-(const float3& a, const float3& b);
__device__ float3 operator*(const float3& a, const float3& b);
__device__ float3 operator/(const float3& a, const float3& b);