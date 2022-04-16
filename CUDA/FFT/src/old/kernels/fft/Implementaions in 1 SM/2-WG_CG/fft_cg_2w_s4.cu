#define N 512
#define M 2*N
#define D 2*N   
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SC[M], SROT[N];
	short j = threadIdx.x;
	short n = logf(N) / logf(2);

	SROT[j / 2] = ROT[j / 2];

	SA[j] = A[j];
	__syncthreads();


	short i = j / 2;
	short k = j % 2;
	short l = i / 2;
	short m = i % 2;
	short sign0 = (-2) * k + 1;
	short sign1 = m * (-2) + 1;

	//short inx0 = ind2 + (!m)*2;
	//short q = j%4;
	//short ind0 = l<<1;
	//short ind1 = (l<<1) + N;
	//short ind2 = (l<<2) + q;//=j

	for (short s = 1; s <= n; s++)
	{
		short r0 = (l / (1 << (s - 1))) * (1 << (s - 1));
		SB[j] = SA[(l << 1) + k] + sign1 * SA[(l << 1) + N + k];
		SC[j] = SB[j + (!m) * 2] * SROT[(r0 << 1) + m];
		SA[j] = (!m) * SB[j] + m * (SC[(l << 2) + k] + sign0 * SC[(l << 2) + 2 + (!k)]);
		__syncthreads();
	}


	A[j] = SA[j];

}

#include  <stdio.h> 
#include  <math.h>
int  main()
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

	//__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
	dim3 gridDim(1, 1);
	dim3 blockDim(D, 1);
	fft << <gridDim, blockDim >> > (Ad, ROTd);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}







