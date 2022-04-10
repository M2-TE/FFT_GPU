#define N 1024
#define M 2*N
#define D 2*N   
#define n 10

__global__ void fft(float* A, float* ROT)
{
	clock_t start = clock();


	__shared__ float SA[M], SB[M], SROT[M];
	short j = threadIdx.x;

	SROT[j] = ROT[j];
	SA[j] = A[j];
	__syncthreads();

	short i = j >> 1;
	short k = j % 2;
	short l = i >> 1;
	short m = i % 2;
	short r0 = m * (i >> (n - 1));
	short signk = -(k << 1) + 1;
	short signm = -(m << 1) + 1;
	short x = r0 * (signk);
	short signkr0 = -2 * ((!k) * r0) + 1;
	short ind0 = l << 1;
	short ind1 = (l << 1) + N;
	short s;

	//stage1:
	SB[j + x] = signkr0 * (SA[ind0 + k] + signm * SA[ind1 + k]);
	__syncthreads();
	//stage2:
	s = 2;
	short r1 = (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SA[j] = SB[ind0 + k] + signm * SB[ind1 + k];
	__syncthreads();
	SB[j] = signk * SA[i << 1] * SROT[(r1 << 1) + k] + SA[(i << 1) + 1] * SROT[(r1 << 1) + (!k)];
	__syncthreads();
	//stage3:

	SA[j + x] = signkr0 * (SB[ind0 + k] + signm * SB[ind1 + k]);
	__syncthreads();
	//stage4:
	s = 4;
	short r3 = (1 << (s - 2)) * (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SB[j] = SA[ind0 + k] + signm * SA[ind1 + k];
	__syncthreads();
	SA[j] = signk * SB[i << 1] * SROT[(r3 << 1) + k] + SB[(i << 1) + 1] * SROT[(r3 << 1) + (!k)];
	__syncthreads();

	//stage5:

	//SB[j+x] = signkr0*(SA[ind0+k] + (signm)*SA[ind1+k]);

	SB[j + x] = signkr0 * (SA[ind0 + k] + signm * (SA[ind1 + k]));
	__syncthreads();
	//stage6:
	s = 6;
	short r5 = (1 << (s - 2)) * (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SA[j] = SB[ind0 + k] + signm * SB[ind1 + k];
	__syncthreads();
	SB[j] = signk * SA[i << 1] * SROT[(r5 << 1) + k] + SA[(i << 1) + 1] * SROT[(r5 << 1) + (!k)];
	__syncthreads();

	//stage7:
	SA[j + x] = signkr0 * (SB[ind0 + k] + signm * (SB[ind1 + k]));
	__syncthreads();

	//stage8:
	s = 8;
	short r7 = (1 << (s - 2)) * (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SB[j] = SA[ind0 + k] + signm * SA[ind1 + k];
	__syncthreads();
	SA[j] = signk * SB[i << 1] * SROT[(r7 << 1) + k] + SB[(i << 1) + 1] * SROT[(r7 << 1) + (!k)];
	__syncthreads();

	//stage9:
	SB[j + x] = signkr0 * (SA[ind0 + k] + signm * (SA[ind1 + k]));
	__syncthreads();

	//stage10:
	SA[j] = SB[ind0 + k] + signm * SB[ind1 + k];


	__syncthreads();

	A[j] = SA[j];

	clock_t stop = clock();
	printf("%d \n", stop - start);
}

void DoA()
{
	float A[2 * N];
	float* Ad;
	float ROT[2 * N];
	float* ROTd;

	int memsize = 2 * N * sizeof(float);
	int rotsize = 2 * N * sizeof(float);


	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}
	for (int j = 0; j < (N); j++)
	{

		ROT[2 * j] = cosf((j * (6.2857)) / N);
		ROT[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	cudaMalloc((void**)&Ad, memsize);
	cudaMalloc((void**)&ROTd, rotsize);

	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(ROTd, ROT, rotsize, cudaMemcpyHostToDevice);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	fft<<<1, D >>>(Ad, ROTd);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f us\n", time * 1000.0);

	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);

	
	//printf("The  outputs are: \n");
	//for (int l = 0; l < N; l++) printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}

#undef N
#undef M
#undef D  
#undef n










