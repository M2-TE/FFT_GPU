	#define N 2048
	#define M 2*N
	#define D N/2
	__global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[N],SB[N],SROT[N];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	short i = j>>1;
    	short k = j%2;
	SA[j] = A[j+blockIdx.x*(blockDim.x)];
	SA[j+blockDim.x] = A[j+blockIdx.x*(blockDim.x)+N];

	SROT[j] = ROT[j];
	SROT[j+blockDim.x] = ROT[j+blockDim.x];
	__syncthreads();
	//1st stage:
	short ind0 = i<<1;
    	short ind1 = (i<<1) + (N>>1);
    	short ind2 = i<<2;
    	short signk =-(k<<1) + 1;
	short r0 = i+blockIdx.x*(N>>2);
	SB[ind0 + k*D] =  SA[ind0] + signk*SA[ind1];
	SB[ind0 + k*D + 1] = SA[ind0+1] + signk*SA[ind1+1];

	__syncthreads();
	SA[ind0 +k] = SB[ind0+k];
	SA[ind0+D+k] = signk*SB[ind0+D]*SROT[2*r0+k] + SB[ind0+D+1]*SROT[2*r0+(!k)];	
	__syncthreads();
	A[j+blockIdx.x*(N>>1)] = SA[j+(!blockIdx.x)*(N>>1)];
	__syncthreads();
	SA[j+(!blockIdx.x)*(N>>1)] = A[j+(!blockIdx.x)*(N>>1)];
	__syncthreads();

	//2nd stage:
	for(short s=2; s<=n; s++)
	{
	SB[ind2+k] = SA[ind0+k] + SA[ind1+k];
        SB[ind2+2+k] = SA[ind0+k] - SA[ind1+k];
	
	__syncthreads();
	short r0 = (i/(1<<(s-2)))*(1<<(s-1));
	SA[ind2+k] = SB[ind2+k];
        SA[ind2+2+k] = signk*SB[ind2+2]*SROT[2*r0+k] + SB[ind2+3]*SROT[2*r0+(!k)];
	__syncthreads();
	}

	A[j+blockIdx.x*(N)] = SA[j];
	A[j+blockDim.x+blockIdx.x*(N)] = SA[j+blockDim.x];
	
       }

       #include  <stdio.h> 
       #include  <math.h>
       int  main() 
       { 
        
           float A[2*N]; 
           float *Ad;
	   float ROT[N];
	   float *ROTd; 
	  
           int memsize= 2*N * sizeof(float); 
	   int rotsize = N* sizeof(float);

		for(int i=0; i<N; i++)
		{
			A[2*i]  = i;
			A[2*i+1]= i;	
		}
		for(int j=0; j < (N/2); j++)
		{
			
			ROT[2*j]= cosf((j*(6.2857))/N);
			ROT[2*j+1]=sinf((j*(6.2857))/N);	
		}

           cudaMalloc((void**)&Ad, memsize);
	   cudaMalloc((void**)&ROTd, rotsize); 

           cudaMemcpy(Ad, A, memsize,  cudaMemcpyHostToDevice); 
	   cudaMemcpy(ROTd, ROT, rotsize,  cudaMemcpyHostToDevice); 

           //__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(2,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

     }
	




















	
