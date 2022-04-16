#pragma once

// N = input size of FFT
// M = input size of data (N + Imaginary)
// W = word group size
// T = number of threads
// S = number of stages/butterflies
template<uint N, uint M, uint W, uint T, uint S>
__global__ void KernelFFTNew(float* A, float* ROT, uint* pCycles)
{
	START();
	__shared__ float SA[M], SROT[N];

	// TODO: CALC ROTS HERE
	
	// indices
	const uint tid = threadIdx.x;
	//// these are goin to cause bank conflicts!
	//const uint ind0 = 2u * tid; // word 1
	//const uint ind1 = 2u * tid + N; // word 2
	//const uint ind2 = 4u * tid; // output

	const uint tid2 = tid * 2u;
	uint2 in_real = make_uint2(tid, tid + T); // real parts of words 1 and 2
	uint2 out_real = make_uint2(tid2, tid2 + 1); // real (bank conflicts)

	// global mem -> shared mem
	SROT[tid] = ROT[tid];
	SROT[tid + blockDim.x] = ROT[tid + blockDim.x];
	SA[tid] = A[tid];
	SA[tid + blockDim.x] = A[tid + blockDim.x];
	SA[tid + 2 * blockDim.x] = A[tid + 2 * blockDim.x];
	SA[tid + 3 * blockDim.x] = A[tid + 3 * blockDim.x];


#pragma unroll // can unroll because S is known at compile time
	for (uint s = 1; s <= S; s++)
	{
		// make sure shared memory is written across all threads
		__syncthreads();

		float sb0 = SA[in_real.x] + SA[in_real.y];
		float sb1 = SA[in_real.x + N] + SA[in_real.y + N];

		float sb2 = SA[in_real.x] - SA[in_real.y];
		float sb3 = SA[in_real.x + N] - SA[in_real.y + N];

		// rotation index (?) based on stage
		uint r0 = (tid / (1 << (s - 1))) * (1 << (s - 1));

		SA[out_real.x] = sb0;
		SA[out_real.x + N] = sb1;
		SA[out_real.y] = sb2 * SROT[2 * r0] + sb3 * SROT[2 * r0 + 1];
		SA[out_real.y + N] = -sb2 * SROT[2 * r0 + 1] + sb3 * SROT[2 * r0];
	}

	__syncthreads();
	A[tid] = SA[tid];
	A[tid + blockDim.x] = SA[tid + blockDim.x];
	A[tid + 2 * blockDim.x] = SA[tid + 2 * blockDim.x];
	A[tid + 3 * blockDim.x] = SA[tid + 3 * blockDim.x];

	STOP();
	WRITE_CYCLES(tid);
}
