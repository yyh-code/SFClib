
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <cmath>
#include <time.h>

//using namespace std;
//#define dimension_size 32

__global__ void add(unsigned int ** z,int a,int c, int b,int d)
{
	unsigned short pre_row = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned short pre_col = threadIdx.y + blockIdx.y * blockDim.y;
	z[pre_row][pre_col] = 0;//一开始忘记给z[]赋初值！！！
	unsigned short row = pre_row + c;
	unsigned short col = pre_col + b;
	//printf("%d,%d\n", pre_row, pre_col);
	if (row >=c && row <= a && col >= b && col <= d) {
		for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
			z[pre_row][pre_col] |= (row & 1U << i) << (i + 1) | (col & 1U << i) << i;
		}
	}
}
void encode()
{
	printf("coordinate input:\n");
	unsigned int row, col,z=0;
	scanf("%d%d", &row, &col);
	for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
		z|= (row & 1U << i) << (i + 1) | (col & 1U << i) << i;
	}
	printf("%d\n", z);
}
void decode()
{
	unsigned int z;
	int flag=0, i=0, j=0;
	int row=0, col=0;
	printf("SFC_value input:\n");
	scanf("%d", &z);
	unsigned int a[16] = { 0 };//行
	unsigned int b[16] = { 0 };//列
	while (z>0)
	{
		if (flag == 0) {
			b[i] = z % 2;
			i = i + 1;
			z = z / 2;
			flag = 1;
		}
		else {
			a[j] = z % 2;
			j = j + 1;
			z = z / 2;
			flag = 0;
		}
	}
	while (j > 0) {
		if (a[--j] == 1) {
			row += pow(2,j);
		}	
	}
	while (i > 0) {
		if (b[--i] == 1) {
			col += pow(2, i);
		}
	}
	printf("%d,%d\n", row, col);
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
	//int number = N * M;
	//int nBytes = number*sizeof(unsigned int);
	unsigned int **z;
	unsigned int *h_c;
	unsigned int **d_z;
	unsigned int *d_c;
	cudaMalloc((void**)&d_z, N * sizeof(unsigned int*));
	cudaMalloc((void**)&d_c, N * M * sizeof(unsigned int));
	z = (unsigned int**)malloc(N * sizeof(unsigned int*));
	h_c = (unsigned int*)malloc(N*M * sizeof(unsigned int));
	for (int r = 0; r < N; r++)
	{
		z[r] = d_c + r*M;
	}


	cudaMemcpy((void*)d_z, (void*)z, N * sizeof(unsigned int*), cudaMemcpyHostToDevice);
	int BLOCKCOLS = 1;
	int BLOCKROWS = 1;
	int gridCols = (M + BLOCKCOLS - 1) / BLOCKCOLS;
	int gridRows = (N + BLOCKROWS - 1) / BLOCKROWS;
	dim3 gridSize(gridCols, gridRows);
	dim3 blockSize(BLOCKCOLS, BLOCKROWS);

	//dim3 blockSize(1,16);
	//dim3 gridSize((number + blockSize.x*blockSize.y - 1) / (blockSize.x*blockSize.y));
	//add << <gridSize, blockSize >> >(d_z);
	add << <gridSize, blockSize >> >(d_z, a, c, b, d);
	//add << <1, blockSize >> >(d_z, a, c, b, d);
	cudaMemcpy((void*)h_c, (void*)d_c, N*M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	FILE *outfile;
	outfile = fopen("SFC_z.txt", "w");
	if (outfile == NULL) {
		printf("无法打开文件\n");
	}
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < M; col++)
		{
			fprintf(outfile, "%d ", h_c[row*M + col]);
		}
		fprintf(outfile, "\n");
	}
	fclose(outfile);
	printf("finished!\n");
	clock_t end_time = clock();
	float clockTime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;
	printf("Running time is:   %3.2f ms\n", clockTime);
	cudaFree(d_z);
	cudaFree(d_c);
	free(z);
	free(h_c);
}
int main()
{
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
	return 0;
}
