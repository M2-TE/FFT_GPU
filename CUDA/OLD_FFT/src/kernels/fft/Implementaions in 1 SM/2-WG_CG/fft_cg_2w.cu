#define N 1024
#define M 2*N
#define D N/2

#include  <stdio.h> 
#include  <math.h>

__global__  void fft(float* A, float* ROT)
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

int main()
{
	float A[2 * N];
	float* Ad;
	float ROT[N];
	float* ROTd;

	int memsize = 2 * N * sizeof(float);
	int rotsize = N * sizeof(float);

	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}

	for (int j = 0; j < (N / 2); j++)
	{
		ROT[2 * j] = cosf((j * (6.2857)) / N);
		ROT[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	cudaMalloc((void**)&Ad, memsize);
	cudaMalloc((void**)&ROTd, rotsize);

	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(ROTd, ROT, rotsize, cudaMemcpyHostToDevice);

	// __global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
	dim3 gridDim(1, 1);
	dim3 blockDim(D, 1);
	fft << <gridDim, blockDim >> > (Ad, ROTd);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);
}
