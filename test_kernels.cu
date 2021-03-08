__global__ void matmul(int n, const float *A, const float *B, float *C){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  if(row < n && col < n){
    float val = 0.0;
    for(int i=0; i<n; ++i){
      val += A[row*n + i]*B[n*i + col];
    }
    C[row*n + col] = val;
  }
}


__global__ void addone(int n_cols, float *A)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//int bx = blockIdx.x;
	//int by = blockIdx.y;
	
	int idx = tx*n_cols + ty;
	float val = tx*n_cols + ty + 1.0;
	//float val = bx*n_cols + by + 1.0;
	
	A[idx] = val;
}

