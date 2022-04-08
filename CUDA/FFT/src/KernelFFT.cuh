#pragma once

typedef unsigned int uint; // custom type for readability
template<uint nInput, uint nWordsPerThread, uint nStages> // TOOD: find a way to do the logf(N) / logf(nWordsPerThread) at compile time!
__global__ void KernelFFT(float* A, float* ROT)
{
	// TODO: check if these have an impact at runtime or if they actually work like c++ static constexpr
	// These might not work with earlier CUDA versions, so probably better off putting everything into template parameters instead
	constexpr uint N = nInput;
	constexpr uint M = nInput * 2u;
	constexpr uint W = nWordsPerThread; // word group
	constexpr uint T = M / W; // total number of threads (currently same as blockDim.x)
	//short n = logf(N) / logf(2); // n is amount of stages, 2 is word size here TODO: MAKE THIS CONSTEXPR
	constexpr short n = nStages; // TODO: turn into uint

	__shared__ float SA[M], SB[M], SROT[N];


	// replace all the short with uint!
	short j = threadIdx.x; // where is threadIdx stored? may not need to take up an extra register

	SROT[j] = ROT[j]; // word[0] rotation
	SROT[j + blockDim.x] = ROT[j + blockDim.x]; // word[1] rotation

	// transfer input data to shared memory
	// TODO: can probably make do with only one (two total) read, since first input on word[0] == word[1]
	SA[j] = A[j]; // word[0] of REAL
	SA[j + blockDim.x] = A[j + blockDim.x]; // word[1] of REAL
	SA[j + 2 * blockDim.x] = A[j + 2 * blockDim.x]; // word[0] of IMAGINARY
	SA[j + 3 * blockDim.x] = A[j + 3 * blockDim.x]; // word[1] of IMAGINARY

	__syncthreads();
	short ind0 = 2 * j;
	short ind1 = 2 * j + N;
	short ind2 = 4 * j;

#pragma unroll // can unroll because n is now compile-time constant
	for (short s = 1; s <= n; s++)
	{
		// take care of which shared memory banks are accessed here, needs close inspection

		SB[ind2] = SA[ind0] + SA[ind1];
		SB[ind2 + 1] = SA[ind0 + 1] + SA[ind1 + 1];
		SB[ind2 + 2] = SA[ind0] - SA[ind1];
		SB[ind2 + 3] = SA[ind0 + 1] - SA[ind1 + 1];

		short r0 = (j / (1 << (s - 1))) * (1 << (s - 1));

		SA[ind2] = SB[ind2];
		SA[ind2 + 1] = SB[ind2 + 1];
		SA[ind2 + 2] = SB[ind2 + 2] * SROT[2 * r0] + SB[ind2 + 3] * SROT[2 * r0 + 1];
		SA[ind2 + 3] = -SB[ind2 + 2] * SROT[2 * r0 + 1] + SB[ind2 + 3] * SROT[2 * r0];
		__syncthreads();
	}

	A[j] = SA[j];
	A[j + blockDim.x] = SA[j + blockDim.x];
	A[j + 2 * blockDim.x] = SA[j + 2 * blockDim.x];
	A[j + 3 * blockDim.x] = SA[j + 3 * blockDim.x];
}
