#include <cufft.h>

float DoCUFFT(uint NX, uint nRepetitions, bool bPrintOutput = false)
{
#define BATCH 1

	cufftHandle plan;
	cufftComplex* data;
	float* input = new float[NX * 2];

	for (int i = 0; i < NX; i++)
	{
		input[2 * i] = i;
		input[2 * i + 1] = i;
	}

	cudaMalloc((void**)&data, sizeof(cufftComplex) * NX);
	cudaMemcpy(data, input, sizeof(cufftComplex) * NX, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

	CUDA_TIMER_START();
	for (uint i = 0u; i < nRepetitions; i++) {
		cufftExecC2C(plan, data, data, CUFFT_FORWARD);
		cudaDeviceSynchronize();
	}
	CUDA_TIMER_END();

	if (bPrintOutput) {
		printf("The  outputs are: \n");
		cudaMemcpy(input, data, sizeof(cufftComplex) * NX, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		//for (int l = 0; l < NX; l++) printf("RE:A[%d]=%.2f\t\t\t, IM: A[%d]=%.2f\t\t\t \n ", 2 * l, input[2 * l], 2 * l + 1, input[2 * l + 1]);
		CUDA_TIMER_PRINT();
	}

	delete input;
	cufftDestroy(plan);
	cudaFree(data);

	return time;
}