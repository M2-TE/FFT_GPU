	#define N 512
	#define M 2*N
	#define D N   
       __global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SROT[N];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	SROT[j] = ROT[j];
	
	SA[j] = A[j];
	SA[j+blockDim.x] = A[j+blockDim.x];
	__syncthreads();
	

	short i = j/2;
	short k = j%2;
	for(short s= 1; s<= n ; s++)
	  {
		short p = (2*N)/(1<<s); 
		short sign = (-2)*k+1;
		short ind_tmp = 2*i + (i/(1<<(n-s)))*(1<<(n-s+1));
		short ind0 = ind_tmp + k*p;
		short index = ind0+sign*p;
		
		SB[ind0] = SA[index] + sign*SA[ind0];
		SB[ind0+1] = SA[index+1] + sign*SA[ind0+1];
		
		short r0 = (i%(1<<(n-s)))*(1<<(s-1));		
		short inx1 = ind0 -k*p +k;
		short inx2 = ind0 +(!k)*p;
		SA[inx1] = SB[inx1];
		SA[inx2+k] = sign*SB[inx2]*SROT[2*r0+k] + SB[inx2+1]*SROT[2*r0+(!k)];
	  }
	
	A[j] = SA[j];
	A[j+blockDim.x] = SA[j+blockDim.x];

    }

       #include  <stdio.h> 
       #include  <math.h>
       int  main() 
       { 
        
           float A[2*N]; 
           float *Ad;
	   float ROT[N];
	   float *ROTd; 
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	  
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

	   cudaEventRecord(start, 0);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
	   cudaEventRecord(stop, 0);
	   cudaEventSynchronize(stop);


           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

	   cudaEventElapsedTime(&time, start, stop);
       	    printf ("Time for the kernel: %f us\n", time*1000.0);

            /*printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); */

     }
	
		 
