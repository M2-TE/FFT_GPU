	#define N 256
	#define M 2*N
	#define D N   
       __global__  void fft(float* A , float* ROT ) 
       { 
	__shared__ float SA[M],SB[M],SROT[N];
	short j   = threadIdx.x;
	short n = logf(N)/logf(4);
	SROT[j] = ROT[j];
	
	SA[j] = A[j];
	SA[j+blockDim.x] = A[j+blockDim.x];
	__syncthreads();
	
	short ind0,r0,r1,sign1,sign2,index1,index2;
	short i = j/2;
	short k = j%2;
	short h = i/2;
	short l = j%4;
	short m = i%2;
		
	for(short s=1; s<=n; s++)
	{
		short p = M/(1<<(2*s));
		short ind_tmp = 2*(h+(h/(1<<2*(n-s)))*(1<<2*(n-s))*3);
		short r_tmp = (h%(1<<2*(n-s)))*(1<<2*(s-1));
				
		ind0 = ind_tmp+l*p;
		
		//stage1:
		sign1= m*(-2)+1;
		r0 = r_tmp + k*(N/4);
		index1 = ind0+sign1*2*p;
				
		SB[ind0] = SA[index1] + sign1*SA[ind0];
		SB[ind0+1] = SA[index1+1] + sign1*SA[ind0+1];
		
		short inx1 = ind0 - m*2*p + m;
		short inx2 = ind0 + (!m)*2*p ; 

		SA[inx1]   = SB[inx1];
		SA[inx2+m] = sign1*SB[inx2]*SROT[2*r0+m] + SB[inx2+1]*SROT[2*r0+(!m)];
		__syncthreads();
		//stage2:
		
		sign2 = k*(-2)+1;
		r1 = r_tmp*2; 
		index2 = ind0+sign2*p;
		
		SB[ind0] = SA[index2] + sign2*SA[ind0];
		SB[ind0+1] = SA[index2+1] + sign2*SA[ind0+1];
			
		short inx3 = ind0 - k*p + k;
		short inx4 = ind0 + (!k)*p ; 
	
		SA[inx3] = SB[inx3];
		SA[inx4+k] = sign2*SB[inx4]*SROT[2*r1+k] + SB[inx4+1]*SROT[2*r1+(!k)];

		/*if(m==0)
		{
			SA[ind0]   = SB[ind0];
			SA[index1] = SB[index1]*SROT[2*r0] + SB[index1+1]*SROT[2*r0+1];
		}
		else
		{
			SA[index1+1] = SB[index1+1];
			SA[ind0+1]   = -SB[ind0]*SROT[2*r0+1] + SB[ind0+1]*SROT[2*r0];
		}*/
	__syncthreads();
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
	
	
		
