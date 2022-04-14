#pragma once

// almost consistent overhead (?):
// Debug: ~311 clock cycles
// Release: ~6 clock cycles
#ifdef _DEBUG
	#define START() clock_t start = clock()
	#define STOP() clock_t stop = clock()
	#define WRITE_CYCLES(tid) pCycles[tid] = (uint)(stop - start);
#else
	#define START() clock_t start = clock()
	#define STOP() clock_t stop = clock()
	#define WRITE_CYCLES(tid) pCycles[tid] = (uint)(stop - start);
#endif

// this is radix 2, not radix 2^2

// N = input size of FFT
// M = input size of data (N + Imaginary)
// W = word group size
// T = number of threads
// S = number of stages/butterflies
typedef unsigned int uint;
template<uint N, uint M, uint W, uint T, uint S>
__global__ void KernelFFT(float* A, float* ROT, uint* pCycles)
{
	START();
	__shared__ float SA[M], SROT[N];

	// DO ROTS HERE

	// indices (these are goind to cause bank conflicts!)
	const uint tid = threadIdx.x;
	const uint ind0 = 2 * tid; // input real (word 1)
	const uint ind1 = 2 * tid + N; // real (word 2)
	const uint ind2 = 4 * tid; // output

	// These coalesce properly.
	//
	// transfer rotations to shared memory
	SROT[tid] = ROT[tid]; // word[0] rotation
	SROT[tid + T] = ROT[tid + T]; // word[1] rotation

	// transfer input data to shared memory
	SA[tid] = A[tid];
	SA[tid + T] = A[tid + T];
	SA[tid + 2 * T] = A[tid + 2 * T];
	SA[tid + 3 * T] = A[tid + 3 * T];

	// make sure shared memory is written across all threads
	__syncthreads();

#pragma unroll // can unroll because S is known at compile time
	for (uint s = 1; s <= S; s++)
	{
		// could store the ind0 + 1 in a register so it doesnt get calculated again each iteration?
		// bank conflict!
		float sb0 = SA[ind0] + SA[ind1]; // real parts addition.
		float sb1 = SA[ind0 + 1] + SA[ind1 + 1]; //img parts addition.

		float sb2 = SA[ind0] - SA[ind1]; //real parts subtraction.
		float sb3 = SA[ind0 + 1] - SA[ind1 + 1]; //img parts subtraction.

		// rotation index (?) based on stage
		uint r0 = (tid / (1 << (s - 1))) * (1 << (s - 1));

		SA[ind2] = sb0;
		SA[ind2 + 1] = sb1;
		SA[ind2 + 2] = sb2 * SROT[2 * r0] + sb3 * SROT[2 * r0 + 1];
		SA[ind2 + 3] = -sb2 * SROT[2 * r0 + 1] + sb3 * SROT[2 * r0];
		__syncthreads();
	}


	// write to output (dram)
	A[tid] = SA[tid];
	A[tid + T] = SA[tid + T];
	A[tid + 2 * T] = SA[tid + 2 * T];
	A[tid + 3 * T] = SA[tid + 3 * T];

	STOP();
	WRITE_CYCLES(tid);
}
