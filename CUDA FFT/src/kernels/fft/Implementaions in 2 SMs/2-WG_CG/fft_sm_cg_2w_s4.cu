	#define N 1024
	#define M 2*N
	#define D N 

	__global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[N],SB[N],SC[N],SROT[M];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	short i =j>>1;
	SA[i] = A[i+blockIdx.x*(N>>1)];
	SA[i+(N>>1)] = A[i+blockIdx.x*(N>>1)+N];

	SROT[j] = ROT[j];
	

	__syncthreads();

	short k = j%2;
	short l = i>>1;
	short m = i%2;
	short signk = -(k<<1) + 1;
	short signm =  -(m<<1) +1;
	short ind0 = l<<1;
	short ind1 = (l<<1)+(N>>1);
	short r0 = l+blockIdx.x*(N>>2);
	SB[ind0+m*(D>>1)+k] = SA[ind0+k] + signm*SA[ind1+k];
	SC[j] = SB[ind0+(D>>1)+k]*SROT[(r0<<1)+m];
	SA[ind0+m*(D>>1)+k] = (!m)*SB[ind0+m*(D>>1)+k] + m*(SC[(l<<2)+k]+signk*SC[(l<<2)+2+(!k)]);
	
	__syncthreads();
	A[i+blockIdx.x*(N>>1)] = SA[i+(!blockIdx.x)*(N>>1)];
	__syncthreads();
	SA[i+(!blockIdx.x)*(N>>1)] = A[i+(!blockIdx.x)*(N>>1)];
	__syncthreads();
	for(short s=2; s<=n; s++)
	{
	SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	short r0 = (l/(1<<(s-2)))*(1<<(s-1));
	SC[j] = SB[j+(!m)*2]*SROT[(r0<<1)+m];
	SA[j] = (!m)*SB[j] + m*(SC[(l<<2)+k]+signk*SC[(l<<2)+2+(!k)]);

	}

	A[i+blockIdx.x*(N)] = SA[i];
	A[i+(N>>1)+blockIdx.x*(N)] = SA[i+(N>>1)];
	

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
	

