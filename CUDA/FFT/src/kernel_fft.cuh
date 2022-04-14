#pragma once


// N = input size of FFT
// M = input size of data (N + Imaginary)
// W = word group size
// T = number of threads
// S = number of stages/butterflies
template<uint N, uint M, uint W, uint T, uint S>
__global__ void KernelFFT(float* A, float* ROT, uint* pCycles)
{
	START();
	__shared__ float SA[M], SROT[N];

	// TODO: CALC ROTS HERE

	// indices
	const uint tid = threadIdx.x;
	const uint iRot = tid * 2u;
	const uint iInOut = tid * 4u;
	// these are goin to cause bank conflicts!
	const uint ind0 = 2u * tid; // input real (word 1)
	const uint ind1 = 2u * tid + N; // real (word 2)
	const uint ind2 = 4u * tid; // output

	// transfer rotations to shared memory
	SROT[iRot + 0] = ROT[iRot + 0];
	SROT[iRot + 1] = ROT[iRot + 1];

	// transfer input data to shared memory
	SA[iInOut + 0] = A[iInOut + 0];
	SA[iInOut + 1] = A[iInOut + 1];
	SA[iInOut + 2] = A[iInOut + 2];
	SA[iInOut + 3] = A[iInOut + 3];

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
	A[iInOut + 0] = SA[iInOut + 0];
	A[iInOut + 1] = SA[iInOut + 1];
	A[iInOut + 2] = SA[iInOut + 2];
	A[iInOut + 3] = SA[iInOut + 3];

	STOP();
	WRITE_CYCLES(tid);
}
