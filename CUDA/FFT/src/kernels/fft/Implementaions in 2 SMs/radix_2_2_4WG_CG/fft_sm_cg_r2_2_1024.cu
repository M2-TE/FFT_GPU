	#define N 1024
	#define M 2*N
	#define D N/2
	__global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[N],SB[N],SROT[M];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	SA[j] = A[j+blockIdx.x*(N>>1)];
	SA[j+blockDim.x] = A[j+blockIdx.x*(N>>1)+N];
	SROT[j] = ROT[j];
	SROT[j+blockDim.x] = ROT[j+blockDim.x];
	SROT[j+2*blockDim.x] = ROT[j+2*blockDim.x];
	SROT[j+3*blockDim.x] = ROT[j+3*blockDim.x];
	__syncthreads();
	short g = j+blockIdx.x*blockDim.x;
	 short i = j>>1;
    	short k = j%2;
   
    	short ind0 = i<<1;
    	short ind1 = (i<<1) + (N>>1);
    	short ind2 = j<<1;
    	short signk =-(k<<1) + 1;
    	short s;
	//stage1:
	
	short r0 = (g%2)*(g>>(n-1));//(g>>(n-1))*((g>>(n-2))%2);
	short signr0 = -(r0<<1) + 1;
	SB[ind0 + k*D + r0] = signr0*(SA[ind0] + signk*SA[ind1]);
	SB[ind0 + k*D +(!r0)] = SA[ind0+1] + signk*SA[ind1+1];

	//SB[ind0 + k*D ] = r0;
	//SB[ind0 + k*D +1] = r0;
	__syncthreads();
	A[j+blockIdx.x*(N>>1)] = SB[j+(!blockIdx.x)*(N>>1)];
	__syncthreads();
	SB[j+(!blockIdx.x)*(N>>1)] = A[j+(!blockIdx.x)*(N>>1)];
	__syncthreads();

	 //stage2:
	s=2;
	
	short r1 = (((g%2)<<1) + ((g>>(n-s+1))%2))*((g%(1<<(n-s+1)))>>1);

	SA[ind2] = SB[ind0] + signk*SB[ind1];
	SA[ind2+1] = SB[ind0+1] + signk*SB[ind1+1];
	//__syncthreads();
	SB[ind2] = SA[ind2]*SROT[r1<<1] + SA[ind2+1]*SROT[(r1<<1)+1];
	SB[ind2+1] = -SA[ind2]*SROT[(r1<<1)+1] + SA[ind2+1]*SROT[r1<<1];
	__syncthreads();
	//stage3:

	short r2= k*(j>>(n-2));
	short signr2 = -(r2<<1) + 1;
	SA[ind2 + r2] = signr2*(SB[ind0] + signk*SB[ind1]);
	SA[ind2 + (!r2)] = SB[ind0+1] + signk*SB[ind1+1];
	__syncthreads();
	//stage4:

	s=4;
	short r3 = (1<<(s-2))*(((j%2)<<1) + ((j>>1)%2))*(j>>(s-1));
	SB[ind2] = SA[ind0] + signk*SA[ind1];
	SB[ind2+1] = SA[ind0+1] + signk*SA[ind1+1];
	SA[ind2] = SB[ind2]*SROT[r3<<1] + SB[ind2+1]*SROT[(r3<<1)+1];
	SA[ind2+1] = -SB[ind2]*SROT[(r3<<1)+1] + SB[ind2+1]*SROT[r3<<1];
	__syncthreads();
	//stage5:
	SB[ind2 + r2] = signr2*(SA[ind0] + (-(k*2)+1)*SA[ind1]);
	SB[ind2 + (!r2)] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();
	//stage6:

	s=6;
	short r5 = (1<<(s-2))*(((j%2)<<1) + ((j>>1)%2))*(j>>(s-1));
	SA[ind2] = SB[ind0] + signk*SB[ind1];
	SA[ind2+1] = SB[ind0+1] + signk*SB[ind1+1];
	SB[ind2] = SA[ind2]*SROT[r5<<1] + SA[ind2+1]*SROT[(r5<<1)+1];
	SB[ind2+1] = -SA[ind2]*SROT[(r5<<1)+1] + SA[ind2+1]*SROT[r5<<1];
	__syncthreads();

	//stage7:
	SA[ind2 + r2] = signr2*(SB[ind0] + (-(k*2)+1)*SB[ind1]);
	SA[ind2 + (!r2)] = SB[ind0+1] + signk*SB[ind1+1];
	__syncthreads();
	//stage8:

	s=8;
	short r7 = (1<<(s-2))*(((j%2)<<1) + ((j>>1)%2))*(j>>(s-1));
	SB[ind2] = SA[ind0] + signk*SA[ind1];
	SB[ind2+1] = SA[ind0+1] + signk*SA[ind1+1];
	SA[ind2] = SB[ind2]*SROT[r7<<1] + SB[ind2+1]*SROT[(r7<<1)+1];
	SA[ind2+1] = -SB[ind2]*SROT[(r7<<1)+1] + SB[ind2+1]*SROT[r7<<1];
	__syncthreads();


	//stage9:
	SB[ind2 + r2] = signr2*(SA[ind0] + (-(k*2)+1)*SA[ind1]);
	SB[ind2 + (!r2)] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();
	//stage10:
	SA[ind2] = SB[ind0] + signk*SB[ind1];
	SA[ind2+1] = SB[ind0+1] + signk*SB[ind1+1];

	//SB[ind2] = SA[ind0] + signk*SA[ind1];
	//SB[ind2+1] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();

	A[j+blockIdx.x*(N)] = SA[j];
	A[j+blockDim.x+blockIdx.x*(N)] = SA[j+blockDim.x];

      }

       #include  <stdio.h> 
       #include  <math.h>
       int  main() 
       { 
        
           float A[2*N]; 
           float *Ad;
	   float ROT[2*N];
	   float *ROTd; 
	  
           int memsize= 2*N * sizeof(float); 
	   int rotsize = 2*N* sizeof(float);


		for(int i=0; i<N; i++)
		{
			A[2*i]  = i;
			A[2*i+1]= i;	
		}
		for(int j=0; j < (N); j++)
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

