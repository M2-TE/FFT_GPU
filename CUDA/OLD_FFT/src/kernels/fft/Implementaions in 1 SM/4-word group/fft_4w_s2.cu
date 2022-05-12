        #define N 16
	#define M 2*N
	#define D N/2   
       __global__  void fft(float* A , float* ROT ) 
       { 
	__shared__ float SA[M],SB[M],SROT[N];
	short j   = threadIdx.x;
	short n = logf(N)/logf(4);

	SROT[j] = ROT[j];
	SROT[j+blockDim.x] = ROT[j+blockDim.x];
	
	SA[j] = A[j];
	SA[j+blockDim.x] = A[j+blockDim.x];
	SA[j+2*blockDim.x] = A[j+2*blockDim.x];
	SA[j+3*blockDim.x] = A[j+3*blockDim.x];

	__syncthreads(); 
	
	short ind0, ind1;
	short r0,r1;
	short i = j/2;
	short k = j%2;
	for(short s=1; s<=n; s++)
	{
		short p = M/(1<<(2*s));
		short ind_tmp = 2*(i+(i/(1<<2*(n-s)))*(1<<2*(n-s))*3);
		short r_tmp = (i%(1<<2*(n-s)))*(1<<2*(s-1));

		//stage1:
		ind0 = ind_tmp + k*p;
		ind1 = ind0+2*p;
		r0 = r_tmp + k*(N/4);

		SB[ind0]   =  SA[ind0] + SA[ind1];
		SB[ind0+1] =  SA[ind0+1] + SA[ind1+1]; 
		SB[ind1]   =  SA[ind0] - SA[ind1];
		SB[ind1+1] =  SA[ind0+1] - SA[ind1+1];
		
		SA[ind0]   =  SB[ind0];
		SA[ind0+1] =  SB[ind0+1];
		SA[ind1]   =  SB[ind1]*SROT[2*r0] + SB[ind1+1]*SROT[2*r0+1];
		SA[ind1+1] =  -SB[ind1]*SROT[2*r0+1] + SB[ind1+1]*SROT[2*r0]; 
		__syncthreads();
		//stage2:
		short tmp_s2 = ind0 + k*p;
		r1 = 2*r_tmp;
		
		SB[tmp_s2]   =  SA[tmp_s2] + SA[tmp_s2+p];
		SB[tmp_s2+1] =  SA[tmp_s2+1] + SA[tmp_s2+p+1]; 
		SB[tmp_s2+p]   =  SA[tmp_s2] - SA[tmp_s2+p];
		SB[tmp_s2+p+1] =  SA[tmp_s2+1] - SA[tmp_s2+p+1];

		SA[tmp_s2]   =  SB[tmp_s2];
		SA[tmp_s2+1] =  SB[tmp_s2+1];
		SA[tmp_s2+p]   =  SB[tmp_s2+p]*SROT[2*r1] + SB[tmp_s2+p+1]*SROT[2*r1+1];
		SA[tmp_s2+p+1] =  -SB[tmp_s2+p]*SROT[2*r1+1] + SB[tmp_s2+p+1]*SROT[2*r1]; 
	}

		A[j] = SA[j];
		A[j+blockDim.x] = SA[j+blockDim.x];
		A[j+2*blockDim.x] = SA[j+2*blockDim.x];
		A[j+3*blockDim.x] = SA[j+3*blockDim.x];

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

           // __global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(1,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

     }
	

