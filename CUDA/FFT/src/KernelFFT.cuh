#pragma once

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

typedef unsigned int uint;
template<uint N, uint M, uint W, uint T, uint S>
static __global__ void KernelFFT(float* A, float* ROT, uint* pCycles)
{
	__shared__ float SA[M], SROT[N];
	// indices
	const uint tid = threadIdx.x;
	const uint ind0 = 2 * tid; // input real
	const uint ind1 = 2 * tid + N; // input imaginary(?)
	const uint ind2 = 4 * tid; // output

	// transfer rotations to shared memory
	SROT[tid] = ROT[tid]; // word[0] rotation
	SROT[tid + blockDim.x] = ROT[tid + blockDim.x]; // word[1] rotation

	// transfer input data to shared memory
	SA[tid] = A[tid];
	SA[tid + blockDim.x] = A[tid + blockDim.x];
	SA[tid + 2 * blockDim.x] = A[tid + 2 * blockDim.x];
	SA[tid + 3 * blockDim.x] = A[tid + 3 * blockDim.x];

	// make sure shared memory is written across all threads
	__syncthreads();

//#pragma unroll
	for (uint s = 1; s <= S; s++)
	{
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
	A[tid + blockDim.x] = SA[tid + blockDim.x];
	A[tid + 2 * blockDim.x] = SA[tid + 2 * blockDim.x];
	A[tid + 3 * blockDim.x] = SA[tid + 3 * blockDim.x];

	START();
	STOP();
	WRITE_CYCLES(tid);
}

template<uint N, uint M, uint W, uint T, uint S> // TOOD: find a way to do the logf(N) / logf(nWordsPerThread) at compile time!
static __global__ void KernelFFTNotes(float* A, float* ROT)
{
	// almost consistent overhead (?):
	// Debug: ~311 clock cycles
	// Release: ~6 clock cycles
	//START();
	//STOP();

	// TODO: check if these have an impact at runtime or if they actually work like c++ static constexpr
	// These might not work with earlier CUDA versions, so probably better off putting everything into template parameters instead
	//constexpr uint N = nInput; // FFT size
	//constexpr uint M = nInput * 2u; // Data size
	//constexpr uint W = nWordsPerThread; // word group size
	//constexpr uint T = M / W; // total number of threads (currently same as blockDim.x)
	//uint n = logf(N) / logf(2); // n is amount of stages, 2 is word size here TODO: MAKE THIS CONSTEXPR
	//constexpr uint n = nStages; // TODO: turn into uint

	__shared__ float SA[M], /*SB[M],*/ SROT[N]; // whats the point of SB here?


	// replace all the uint with uint!
	uint tid = threadIdx.x; // where is threadIdx stored? may not need to take up an extra register
	const uint ind0 = 2 * tid; // input real
	const uint ind1 = 2 * tid + N; // input imaginary(?)
	const uint ind2 = 4 * tid; // output

	// TODO: TRANSFERS: can probably make this dynamic based on word size?
	// transfer rotations to shared memory
	SROT[tid] = ROT[tid]; // word[0] rotation
	SROT[tid + blockDim.x] = ROT[tid + blockDim.x]; // word[1] rotation

	// transfer input data to shared memory
	// TODO: can probably make do with only one (two total) read, since first input on word[0] == word[1]
	SA[tid] = A[tid];
	SA[tid + blockDim.x] = A[tid + blockDim.x];
	SA[tid + 2 * blockDim.x] = A[tid + 2 * blockDim.x];
	SA[tid + 3 * blockDim.x] = A[tid + 3 * blockDim.x];

	// make sure shared memory is written across all threads
	//START();
	__syncthreads();
	//STOP();


	// TOOD: measure precise clock cycles!
//#pragma unroll // can unroll because S is now compile-time constant
	for (uint s = 1; s <= S; s++)
	{
		// take care of which shared memory banks are accessed here, needs close inspection
		// -> more than 1 word per group results in bank 
		// removed SB usage
		// removed some shared memory reads using extra registers

		// these arent multiple reads, compiler optimizes it
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
	A[tid + blockDim.x] = SA[tid + blockDim.x];
	A[tid + 2 * blockDim.x] = SA[tid + 2 * blockDim.x];
	A[tid + 3 * blockDim.x] = SA[tid + 3 * blockDim.x];
}
