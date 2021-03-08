__global__ void deriv_entropy(int n_train, int n_classes, 
			float* targets, float* sigma_o, float* d_entropy)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int stride = blockDim.x;
	int idx;
	
	for(idx=bx*n_classes+tx; idx<n_train*n_classes; idx+=stride)
	{
		if(idx < n_train*n_classes)
			d_entropy[idx] = -targets[idx] / sigma_o[idx];
	}
}


