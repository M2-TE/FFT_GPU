#pragma once

// this is the 2-word constant geometry version
// NOTE: need to rebuild project to detect changes in .cu files under certain conditions

// TOOD: add wordsPerThread as parameter
template<int N, int M, int D>
__global__ void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SROT[N];
	short j = threadIdx.x;
	short n = logf(N) / logf(2);

	SROT[j] = ROT[j];
	SROT[j + blockDim.x] = ROT[j + blockDim.x];

	SA[j] = A[j];
	SA[j + blockDim.x] = A[j + blockDim.x];
	SA[j + 2 * blockDim.x] = A[j + 2 * blockDim.x];
	SA[j + 3 * blockDim.x] = A[j + 3 * blockDim.x];

	__syncthreads();
	short ind0 = 2 * j;
	short ind1 = 2 * j + N;
	short ind2 = 4 * j;

	for (short s = 1; s <= n; s++)
	{

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

template<int N>
void ExecuteFFT()
{
	static constexpr int nBlocks = 1;
	static constexpr int nThreads = N / 2;
	static constexpr int nData = 2 * N; // size of input/output array
	static constexpr size_t dataWidth = sizeof(float) * nData;
	static constexpr size_t rotsWidth = sizeof(float) * N;

	float* pData = new float[N * 2];
	float* pRots = new float[N];
	float* pdData, *pdRots; // pointers to device ram (dram on GPU)

	//input elements of N-point FFT.
	for (int i = 0; i < N; i++)
	{
		pData[2 * i] = i;
		pData[2 * i + 1] = i;
	}

	//Rotations of N-point FFT.
	for (int j = 0; j < (N / 2); j++)
	{
		pRots[2 * j] = cosf((j * (6.2857)) / N);
		pRots[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	//Memory allocation in Global memory of Device(GPU).
	cudaMalloc(reinterpret_cast<void**>(&pdData), dataWidth);
	cudaMalloc(reinterpret_cast<void**>(&pdRots), rotsWidth);

	//Copying "input elements" and "rotations" of N-point FFT from CPU to GPU(global memory of GPU(Device)).
	cudaMemcpy(pdData, pData, dataWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(pdRots, pRots, rotsWidth, cudaMemcpyHostToDevice);

	dim3 gridDim(1);
	dim3 blockDim(nThreads);

	//kernel invocation
	fft<N, nData, nThreads>KERNEL_GRID(nBlocks, nThreads)(pdData, pdRots);

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(pData, pdData, dataWidth, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%.2f\t\t\t, IM: A[%d]=%.2f\t\t\t \n ", 2 * l, pData[2 * l], 2 * l + 1, pData[2 * l + 1]);
}
