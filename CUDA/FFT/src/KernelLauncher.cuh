#pragma once

template<int nInput>
void ExecuteFFT()
{
	static constexpr int N = nInput;
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
	KernelFFT<nInput, 2, 10>KERNEL_GRID(nBlocks, nThreads)(pdData, pdRots);

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(pData, pdData, dataWidth, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%.2f\t\t\t, IM: A[%d]=%.2f\t\t\t \n ", 2 * l, pData[2 * l], 2 * l + 1, pData[2 * l + 1]);
}
