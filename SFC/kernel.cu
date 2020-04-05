
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <thrust/sort.h>
#include <vector>

using std::vector;
__global__ void en(unsigned int * z, int N,int M)
{
	unsigned short row = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned short col = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = 0;//一开始忘记给z[]赋初值！！！一定是M！！！
	if (row < N && col < M) {
		for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
			k |= (row & 1U << i) << (i + 1) | (col & 1U << i) << i;
			z[M*row + col] = k;
		}
	}
}
void encode()
{
	int N,M;
	printf("input row col:\n");
	scanf("%d%d", &N,&M);
	printf("In process\n");
	clock_t start_time = clock();
	int number = N * M;
	unsigned int *z;
	unsigned int *d_z;
	cudaMalloc((void**)&d_z, number * sizeof(unsigned int));
	z = (unsigned int*)malloc(number * sizeof(unsigned int));
	cudaMemcpy((void*)d_z, (void*)z, number * sizeof(unsigned int), cudaMemcpyHostToDevice);
	int BLOCKCOLS = 16;
	int BLOCKROWS = 16;
	int gridCols = (M + BLOCKCOLS - 1) / BLOCKCOLS;
	int gridRows = (N + BLOCKROWS - 1) / BLOCKROWS;
	dim3 gridSize(gridRows, gridCols);//行列不能反，否则在核函数中计算行列标记会出错
	dim3 blockSize(BLOCKROWS, BLOCKCOLS);
	en << <gridSize, blockSize >> >(d_z, N,M);
	cudaMemcpy((void*)z, (void*)d_z, N*M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	FILE *outfile;
	outfile = fopen("SFC_z.txt", "w");
	if (outfile == NULL) {
		printf("无法打开文件\n");
	}
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < M; col++)
		{
			fprintf(outfile, "%d ", z[M*row + col]);
		}
		fprintf(outfile, "\n");
	}
	fclose(outfile);
	printf("finished!\n");
	clock_t end_time = clock();
	float clockTime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;
	printf("Running time is:   %3.2f ms\n", clockTime);
	cudaFree(d_z);
	free(z);
}

__global__ void de(unsigned int * z, unsigned int * a, unsigned int * b,int row,int col)
{
	unsigned short pre_row = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned short pre_col = threadIdx.y + blockIdx.y * blockDim.y;
	printf("%d", pre_row);
	int flag = 0;
	int i, j;
	unsigned int m[16] = { 0 };//行
	unsigned int n[16] = { 0 };//列
	while (z[pre_row*row + pre_col]>0)
	{
		if (flag == 0) {
			n[i] = z[pre_row*row + pre_col] % 2;
			i = i + 1;
			z[pre_row*row + pre_col] = z[pre_row*row + pre_col] / 2;
			flag = 1;
		}
		else {
			m[j] = z[pre_row*row + pre_col] % 2;
			j = j + 1;
			z[pre_row*row + pre_col] = z[pre_row*row + pre_col] / 2;
			flag = 0;
		}
	}
	while (j > 0) {
		if (m[--j] == 1) {
			int x = j;
			int mul=1;
			while (x > 0) {
				mul = 2*mul;
				x--;
			}
			a[pre_row] += mul;
		}
	}
	while (i > 0) {
		if (n[--i] == 1) {
			int y = i;
			int mul2 = 1;
			while (y > 0) {
				mul2 = 2 * mul2;
				y--;
			}
			b[pre_col] += mul2;
		}
	}
}

void decode()
{
	int row = 1000;
	int col = 1000;
	int number = row * col;
	unsigned int *z;
	z = (unsigned int*)malloc(number * sizeof(unsigned int));
	unsigned int *a;
	a = (unsigned int*)malloc(row * sizeof(unsigned int));
	unsigned int *b;
	b = (unsigned int*)malloc(col * sizeof(unsigned int));
	unsigned int *d_a;
	cudaMalloc((void**)&d_a, row * sizeof(unsigned int));
	unsigned int *d_b;
	cudaMalloc((void**)&d_b, col * sizeof(unsigned int));
	unsigned int *d_z;
	cudaMalloc((void**)&d_z, number * sizeof(unsigned int));
	
	int i, j;
	FILE *fp;
	char infile[10];
	printf("SFC_value input:\n");
	scanf("%s", infile);
	fp = fopen(infile, "r");
	if (fp == NULL)
	{
		printf("cannot open file\n");
		return;
	}
	for (i = 0; i<row; i++)
	{
		for (j = 0; j<col; j++)
		{
			fscanf(fp, "%d ", &z[i*col + j]);
		}
		fscanf(fp, "\n");
	}
	cudaMemcpy((void*)d_a, (void*)a, row * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_b, (void*)b, col * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_z, (void*)z, number * sizeof(unsigned int), cudaMemcpyHostToDevice);
	int BLOCKCOLS = 16;
	int BLOCKROWS = 16;
	int gridCols = (col + BLOCKCOLS - 1) / BLOCKCOLS;
	int gridRows = (row + BLOCKROWS - 1) / BLOCKROWS;
	dim3 gridSize(gridRows, gridCols);//行列不能反，否则在核函数中计算行列标记会出错
	dim3 blockSize(BLOCKROWS, BLOCKCOLS);
	//dim3 gridSize((number + blockSize.x*blockSize.y - 1) / (blockSize.x*blockSize.y));
	//add << <gridSize, blockSize >> >(d_z);
	de << <gridSize, blockSize >> >(d_z,d_a,d_b,row,col);
	//add << <1, blockSize >> >(d_z, a, c, b, d);
	cudaMemcpy((void*)a, (void*)d_a, row * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)b, (void*)d_b, row * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	FILE *outfile;
	outfile = fopen("decode.txt", "w");
	if (outfile == NULL) {
		printf("无法打开文件\n");
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			fprintf(outfile, "(%d,%d)", a[i],b[j]);
		}
		fprintf(outfile, "\n");
	}

}

__global__ void add(unsigned int * z, int a, int c, int b, int d, int M)
{
	unsigned short pre_row = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned short pre_col = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = 0;//一开始忘记给z[]赋初值！！！一定是M！！！
	unsigned short row = pre_row + c;
	unsigned short col = pre_col + b;
	//printf("%d,%d\n", pre_row, pre_col);
	if (row >= c && row <= a && col >= b && col <= d) {
		for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
			k |= (row & 1U << i) << (i + 1) | (col & 1U << i) << i;
			z[M*pre_row + pre_col] = k;
		}
	}
}

void result(unsigned int * z,int N,int M,int number) {
	FILE *outfile;
	outfile = fopen("Query.txt", "w");
	if (outfile == NULL) {
		printf("无法打开文件\n");
	}
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<M; j++)
		{
			fprintf(outfile, "%d ", z[i*M + j]);
		}
		fprintf(outfile, "\n");
	}
	fprintf(outfile, "\n");
	thrust::sort(z, z + number);//排序
	int max = number - 1;
	vector<unsigned int> rangevec;
	int nstart = 0;
	int i = 0; //current pos
	while (1)
	{
		i++;
		if (i == max)//over the last one
		{
			rangevec.push_back(z[nstart]);
			rangevec.push_back(z[i-1]);
			rangevec.push_back(z[i]);
			break;
		}
		if (z[i] != (z[i - 1] + 1)) //discontinuous
		{
			rangevec.push_back(z[nstart]);
			rangevec.push_back(z[i - 1]);
			nstart = i;
		}
	}
	for (int i = 0; i < rangevec.size(); i++)
	{
		if (i % 2 == 0) {
			fprintf(outfile, "%d", rangevec[i]);
		}
		else {
			fprintf(outfile, "-%d", rangevec[i]);
			fprintf(outfile, " ");
		}
	}
	fclose(outfile);
}
void query()
{
	int a, b, c, d;
	printf("box input:\n");
	scanf("%d%d%d%d", &a, &b, &c, &d);
	printf("In process\n");
	clock_t start_time = clock();
	int N = a - c + 1;
	int M = d - b + 1;
	
	int number = N * M;
	//int nBytes = number*sizeof(unsigned int);
	unsigned int *z;
	unsigned int *d_z;
	cudaMalloc((void**)&d_z, number * sizeof(unsigned int));
	z = (unsigned int*)malloc(number * sizeof(unsigned int));
	cudaMemcpy((void*)d_z, (void*)z, number * sizeof(unsigned int), cudaMemcpyHostToDevice);
	int BLOCKCOLS = 16;
	int BLOCKROWS = 16;
	int gridCols = (M + BLOCKCOLS - 1) / BLOCKCOLS;
	int gridRows = (N + BLOCKROWS - 1) / BLOCKROWS;
	dim3 gridSize(gridRows,gridCols);//行列不能反，否则在核函数中计算行列标记会出错
	dim3 blockSize(BLOCKROWS, BLOCKCOLS);
	add << <gridSize, blockSize >> >(d_z, a, c, b, d,M);
	cudaMemcpy((void*)z, (void*)d_z, N*M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	result(z, N, M, number);
	printf("finished!\n");
	cudaFree(d_z);
	free(z);
}

int main()
{
	//decode();
	//encode();
	query();
	int option;
	while (1)
	{
		printf("1.encode   2.decode  3.query\n");
		printf("please input option：");
		scanf("%d", &option);
		switch (option)
		{
		case 1:encode();
			break;
		case 2:decode();
			break;
		case 3:query();
		}
	}
	//clock_t end_time = clock();
	//float clockTime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;
	//printf("Running time is:   %3.2f ms\n", clockTime);
	return 0;
}
