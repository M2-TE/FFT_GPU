#pragma once

#define START() clock_t start = clock()
#define STOP() clock_t stop = clock(); printf("%d \n", stop - start);
// this is radix 2, not radix 2^2

typedef unsigned int uint; // custom type for readability
template<uint nInput, uint nWordsPerThread, uint nStages> // TOOD: find a way to do the logf(N) / logf(nWordsPerThread) at compile time!
static __global__ void KernelFFT(float* A, float* ROT)
{
	// START();
	// STOP();

	// TODO: check if these have an impact at runtime or if they actually work like c++ static constexpr
	// These might not work with earlier CUDA versions, so probably better off putting everything into template parameters instead
	constexpr uint N = nInput;
	constexpr uint M = nInput * 2u;
	constexpr uint W = nWordsPerThread; // word group
	constexpr uint T = M / W; // total number of threads (currently same as blockDim.x)
	//short n = logf(N) / logf(2); // n is amount of stages, 2 is word size here TODO: MAKE THIS CONSTEXPR
	constexpr short n = nStages; // TODO: turn into uint

	__shared__ float SA[M], /*SB[M],*/ SROT[N]; // whats the point of SB here?


	// replace all the short with uint!
	short j = threadIdx.x; // where is threadIdx stored? may not need to take up an extra register

	// TODO: TRANSFERS: can probably make this dynamic based on word size?
	// transfer rotations to shared memory
	SROT[j] = ROT[j]; // word[0] rotation
	SROT[j + blockDim.x] = ROT[j + blockDim.x]; // word[1] rotation

	// transfer input data to shared memory
	// TODO: can probably make do with only one (two total) read, since first input on word[0] == word[1]
	SA[j] = A[j]; // word[0] of REAL
	SA[j + blockDim.x] = A[j + blockDim.x]; // word[1] of REAL
	SA[j + 2 * blockDim.x] = A[j + 2 * blockDim.x]; // word[0] of IMAGINARY
	SA[j + 3 * blockDim.x] = A[j + 3 * blockDim.x]; // word[1] of IMAGINARY

	// make sure shared memory is written across all threads
	__syncthreads();

	// not yet fully sure what these are EXACTLY, but they seem to be the constant indices
	short ind0 = 2 * j; // input real
	short ind1 = 2 * j + N; // input imaginary(?)
	short ind2 = 4 * j; // output

//#pragma unroll // can unroll because n is now compile-time constant
	for (short s = 1; s <= n; s++)
	{
		// take care of which shared memory banks are accessed here, needs close inspection
		// removed SB usage
		// removed some shared memory reads using extra registers

		// word 1
		float sb0 = SA[ind0] + SA[ind1];
		float sb1 = SA[ind0 + 1] + SA[ind1 + 1];

		// word 2
		float sb2 = SA[ind0] - SA[ind1];
		float sb3 = SA[ind0 + 1] - SA[ind1 + 1];

		// rotation index (?) based on stage
		// can make part of this a constexpr
		short r0 = (j / (1 << (s - 1))) * (1 << (s - 1));

		SA[ind2] = sb0;
		SA[ind2 + 1] = sb1;
		SA[ind2 + 2] = sb2 * SROT[2 * r0] + sb3 * SROT[2 * r0 + 1];
		SA[ind2 + 3] = -sb2 * SROT[2 * r0 + 1] + sb3 * SROT[2 * r0];
		__syncthreads();
	}

	// write to output (dram)
	A[j] = SA[j];
	A[j + blockDim.x] = SA[j + blockDim.x];
	A[j + 2 * blockDim.x] = SA[j + 2 * blockDim.x];
	A[j + 3 * blockDim.x] = SA[j + 3 * blockDim.x];
}
