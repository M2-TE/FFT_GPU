#define N 4096
#define M 2*N
#define D N/4

__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[N], SB[N], SROT[N];
	short j = threadIdx.x;
	short n = logf(N) / logf(2);
	SA[j] = A[j + blockIdx.x * (N / 2)];
	SA[j + blockDim.x] = A[j + blockIdx.x * (N / 2) + blockDim.x];
	SA[j + (N / 2)] = A[j + blockIdx.x * (N / 2) + N];
	SA[j + (N / 2) + blockDim.x] = A[j + blockIdx.x * (N / 2) + N + blockDim.x];

	SROT[j] = ROT[j];
	SROT[j + blockDim.x] = ROT[j + blockDim.x];
	SROT[j + 2 * blockDim.x] = ROT[j + 2 * blockDim.x];
	SROT[j + 3 * blockDim.x] = ROT[j + 3 * blockDim.x];

	//1st stage:
	short r0 = j + blockIdx.x * blockDim.x;
	SB[j << 1] = SA[j << 1] + SA[(j << 1) + (N >> 1)];
	SB[(j << 1) + 1] = SA[(j << 1) + 1] + SA[(j << 1) + (N >> 1) + 1];
	SB[(j << 1) + (N >> 1)] = SA[j << 1] - SA[(j << 1) + (N >> 1)];
	SB[(j << 1) + (N >> 1) + 1] = SA[(j << 1) + 1] - SA[(j << 1) + (N >> 1) + 1];
	__syncthreads();
	SA[j << 1] = SB[j << 1];
	SA[(j << 1) + 1] = SB[(j << 1) + 1];

	SA[(j << 1) + (N / 2)] = SB[(j << 1) + (N / 2)] * SROT[2 * r0] + SB[(j << 1) + (N / 2) + 1] * SROT[2 * r0 + 1];
	SA[(j << 1) + (N / 2) + 1] = -SB[(j << 1) + (N / 2)] * SROT[2 * r0 + 1] + SB[(j << 1) + (N / 2) + 1] * SROT[2 * r0];
	__syncthreads();
	A[j + blockIdx.x * (N >> 1)] = SA[j + (!blockIdx.x) * (N >> 1)];
	A[j + blockIdx.x * (N >> 1) + blockDim.x] = SA[j + (!blockIdx.x) * (N >> 1) + blockDim.x];
	__syncthreads();

	SA[j + (!blockIdx.x) * (N >> 1)] = A[j + (!blockIdx.x) * (N >> 1)];
	SA[j + (!blockIdx.x) * (N >> 1) + blockDim.x] = A[j + (!blockIdx.x) * (N >> 1) + blockDim.x];
	__syncthreads();
	//2nd stage:

	short ind0 = j << 1;
	short ind1 = (j << 1) + (N >> 1);
	short ind2 = j << 2;


	/*SB[(j<<2)]   = SA[j<<1] + SA[(j<<1) + (N>>1)];
	SB[(j<<2)+1] = SA[(j<<1) +1] + SA[(j<<1) + (N>>1)+1];
	SB[(j<<2)+2] = SA[(j<<1)]- SA[(j<<1) + (N>>1)];
	SB[(j<<2)+3] = SA[(j<<1) +1] - SA[(j<<1) + (N>>1)+1];
	__syncthreads();
	SA[(j<<2)]   = SB[(j<<2)];
	SA[(j<<2)+1] = SB[(j<<2)+1];
	SA[(j<<2)+2] = SB[(j<<2)+2]*SROT[(j<<2)] + SB[(j<<2)+3]*SROT[(j<<2)+1];
	SA[(j<<2)+3] = -SB[(j<<2)+2]*SROT[(j<<2)+1] + SB[(j<<2)+3]*SROT[(j<<2)];
	__syncthreads();
	A[j+blockIdx.x*(N>>1)] = SA[j+(!blockIdx.x)*(N>>1)];
	A[j+blockIdx.x*(N>>1)+blockDim.x] = SA[j+(!blockIdx.x)*(N>>1)+blockDim.x];
	__syncthreads();

	SA[j+(!blockIdx.x)*(N>>1)] = A[j+(!blockIdx.x)*(N>>1)];
	SA[j+(!blockIdx.x)*(N>>1)+blockDim.x] = A[j+(!blockIdx.x)*(N>>1)+blockDim.x];
	__syncthreads();*/


	for (short s = 2; s <= n; s++)
	{

		SB[ind2] = SA[ind0] + SA[ind1];
		SB[ind2 + 1] = SA[ind0 + 1] + SA[ind1 + 1];
		SB[ind2 + 2] = SA[ind0] - SA[ind1];
		SB[ind2 + 3] = SA[ind0 + 1] - SA[ind1 + 1];
		__syncthreads();
		short r0 = (j / (1 << (s - 2))) * (1 << (s - 1));

		SA[ind2] = SB[ind2];
		SA[ind2 + 1] = SB[ind2 + 1];
		SA[ind2 + 2] = SB[ind2 + 2] * SROT[2 * r0] + SB[ind2 + 3] * SROT[2 * r0 + 1];
		SA[ind2 + 3] = -SB[ind2 + 2] * SROT[2 * r0 + 1] + SB[ind2 + 3] * SROT[2 * r0];
		__syncthreads();
	}

	A[j + blockIdx.x * (N)] = SA[j];
	A[j + blockDim.x + blockIdx.x * (N)] = SA[j + blockDim.x];
	A[j + 2 * blockDim.x + blockIdx.x * (N)] = SA[j + 2 * blockDim.x];
	A[j + 3 * blockDim.x + blockIdx.x * (N)] = SA[j + 3 * blockDim.x];

}

#include  <stdio.h> 
#include  <math.h>
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
	dim3 gridDim(2, 1);
	dim3 blockDim(D, 1);
	fft << <gridDim, blockDim >> > (Ad, ROTd);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}