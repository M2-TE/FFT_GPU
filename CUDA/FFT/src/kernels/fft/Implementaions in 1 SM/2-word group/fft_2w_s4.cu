	#define N 256
	#define M 2*N
	#define D 2*N   
       __global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SC[M],SROT[N];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	SROT[j/2] = ROT[j/2];

	SA[j] = A[j];
	__syncthreads();

	short i = j/2;
	short k = j%2;
	short l = i/2;
	short m = i%2;

	for(short s= 1; s<= n; s++)
	  {
		short p = (2*N)/(1<<s); 
		short sign0 = k*(-2)+1;
		short sign1 = m*(-2)+1;
		short ind_tmp = 2*l + (l/(1<<(n-s)))*(1<<(n-s+1)); 
		short ind0 = ind_tmp + m*p;
		short r0 = (l%(1<<(n-s)))*(1<<(s-1));
		short index1 = ind0 + sign1*p;
		short inx0 = ind0 + (!m)*p;

		SB[ind0+k] = SA[index1+k] + sign1*SA[ind0+k];
		SC[ind0+k] = SB[inx0+k]*SROT[2*r0+m];
		SA[ind0+k] = (!m)*SB[ind0+k] + m*(SC[index1+k]+sign0*SC[ind0+!k]);
	 }
	__syncthreads();
	
	  A[j] = SA[j];
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
           dim3 gridDim(1,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

     }
		
		


		
