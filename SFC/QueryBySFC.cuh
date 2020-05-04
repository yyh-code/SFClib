#ifndef QUERYBYSFC_H_
#define QUERYBYSFC_H_

//#include "stdafx.h"

#include "Point.h"
#include "Rectangle.h"

//#include "OutputSchema2.h"
//#include "SFCConversion2.h"
#include "SFCConversion.h"

#include <iostream>
#include <vector>
#include <list>
#include <tuple>
#include <queue>
#include <algorithm>
#include <map>
#include <malloc.h>
#include <time.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "tbb/parallel_sort.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

using namespace tbb;
using namespace std;

//#define RETURN_RANGES 40

typedef enum
{
	Morton,
	Hilbert,
} SFCType;

template<typename T>//, int nDims = 2
class TreeNode
{
public:
	int level;  //which level: i-th level
	Point<T> minPoint;//(nDims)
	Point<T> maxPoint;//(nDims)

	int nDims;//= nDims

	void operator=(TreeNode const& other)
	{
		level = other.level;
		minPoint = other.minPoint;
		maxPoint = other.maxPoint;
		nDims = other.nDims;
	}

	TreeNode() :nDims(0), level(0)
	{}

	TreeNode(Point<T> minPoint, Point<T> maxPoint, int lvl) //, nDims, nDims
	{
		STATIC_ASSERT(minPoint.returnSize() == maxPoint.returnSize());

		nDims = minPoint.returnSize();

		this->minPoint = minPoint;
		this->maxPoint = maxPoint;

		this->nDims = minPoint.returnSize();

		this->level = lvl;
	}

	/*
	return the idx-th childnode
	one dim, less than middle is 0, bigger than middle is 1
	0~3 for 2d; upper 2|3----10|11;-----So: YX for 2D, ZYX for 3D, TZYX for 4D
				lower 0|1----00|01 ---------put next dimension before current dimension
	*/
	TreeNode<T> GetChildNode(int idx)
	{
		TreeNode<T> nchild;
		nchild.minPoint = this->minPoint;
		nchild.maxPoint = this->maxPoint;

		nchild.nDims = this->minPoint.returnSize();

		for (int i = 0; i < nDims; i++)
		{
			if ((idx >> i) & 1)  //the bit on the i-th dimension is 1: bigger
			{
				nchild.minPoint[i] = (this->minPoint[i] + this->maxPoint[i]) / 2;
			}
			else  //the bit on the i-th dimension is 0: smaller
			{
				nchild.maxPoint[i] = (this->minPoint[i] + this->maxPoint[i]) / 2;
			}
		}

		nchild.level = this->level + 1;

		return nchild;
	}

	/*
	return the relationship between treenode and queryRectangle
	0: treenode is equal to queryRectangle
	1: treenode contains queryRectangle
	2: treenode intersects queryRectangle
	-1(default): not overlap
	*/
	int Spatialrelationship(Rect<T> qrt)
	{
		/*
		equal:
		if (nrt.x0 == qrt.x0 && nrt.y0 == qrt.y0 &&
		nrt.x1 == qrt.x1 && nrt.y1 == qrt.y1)
		return 0;
		*/
		int ncmp = 1;
		for (int i = 0; i < nDims; i++)
		{
			ncmp &= this->minPoint[i] == qrt.minPoint[i] && this->maxPoint[i] == qrt.maxPoint[i];
		}
		if (ncmp) return 0;

		/*
		fully contain:
		if (nrt.x0 <= qrt.x0 && nrt.y0 <= qrt.y0 &&
		nrt.x1 >= qrt.x1 && nrt.y1 >= qrt.y1)
		return 1;
		*/
		ncmp = 1;
		for (int i = 0; i < nDims; i++)
		{
			ncmp &= this->minPoint[i] <= qrt.minPoint[i] && this->maxPoint[i] >= qrt.maxPoint[i];
		}
		if (ncmp) return 1;

		/*
		intersect:
		//http://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
		RectA.Left < RectB.Right && RectA.Right > RectB.Left && RectA.Top > RectB.Bottom && RectA.Bottom < RectB.Top
		this can be extended more dimensions
		//http://stackoverflow.com/questions/5009526/overlapping-cubes
		if (nrt.x0 < qrt.x1 && nrt.x1 > qrt.x0 &&
		nrt.y0 < qrt.y1 && nrt.y1 > qrt.y0)
		return 2;
		*/
		ncmp = 1;
		for (int i = 0; i < nDims; i++)
		{
			ncmp &= this->minPoint[i] < qrt.maxPoint[i] && this->maxPoint[i] > qrt.minPoint[i];
		}
		if (ncmp) return 2;

		//not overlap
		return -1;
	}
};

template<typename T>//, int nDims = 2, int mBits = 4
class QueryBySFC
{
public:
	int nDims;
	int mBits;
private:
	void query_approximate(TreeNode<T> nd, Rect<T> queryrect, vector<TreeNode<T>>& resultTNode);

	void query_approximate2(TreeNode<T> nd, Rect<T> queryrect, vector<TreeNode<T>>& resultTNode, int nranges, int ktimes);

public:
	vector<sfc_bigint>  RangeQueryByRecursive_LNG_P(Rect<T> queryrect, SFCType sfc_type, int nranges, int ktimes);

	vector<sfc_bigint>  RangeQueryByRecursive_LNG(Rect<T> queryrect, SFCType sfc_type, int nranges, int ktimes);
	QueryBySFC(int dims, int bits) :nDims(dims), mBits(bits)
	{	}


};

///depth-first traversal in the 2^n-ary tree
template<typename T>//, int nDims, int mBits
//void QueryBySFC<T, nDims, mBits>::query_approximate(TreeNode<T> nd, Rect<T> queryrect, vector<TreeNode<T, nDims>>& resultTNode)
void QueryBySFC<T>::query_approximate(TreeNode<T> nd, Rect<T> queryrect, vector<TreeNode<T>>& resultTNode)
{

	/*
	divide current tree node
	*/
	int nary_num = 1 << nDims;  //max count: 2^nDims
	vector<TreeNode<T>> nchild(nary_num);
	/*
	find the currentnode exactly contains queryrectangle; and its child node intersects queryrectangle
	*/
	TreeNode<T> currentNode = nd;
	int res = 1;
	do
	{
		for (int i = 0; i < nary_num; i++)
		{
			nchild[i] = currentNode.GetChildNode(i);
			if (nchild[i].Spatialrelationship(queryrect) == 0)  //equal: stop
			{
				resultTNode.push_back(nchild[i]);
				return;
			}
			else if (nchild[i].Spatialrelationship(queryrect) == 2)  //intersect: divide queryrectangle
			{
				res = 0;
				break;
			}
			else  if (nchild[i].Spatialrelationship(queryrect) == 1)//contain: divide the tree node
			{
				currentNode = nchild[i];
				break;
			}
		}
	} while (res);


	/*
	divide the input query rectangle into even parts, e.g. 2 or 4 parts
	0~3 for 2d; upper 2|3----10|11;----- YX for 2D, ZYX for 3D, TZYX for 4D--each dim one bit
				lower 0|1----00|01 ------one dim: less = 0; greater = 1
	*/

	vector<Rect<T>> qrtcut(nary_num);  //2^nDims parts
	vector<int> qrtpos(nary_num);  //the qrtcut corresponds to treenode
	for (int i = 0; i < nary_num; i++)
	{
		qrtpos[i] = 0;
	}
	vector<int> mid(nDims);  //middle cut line--dim number
	for (int i = 0; i < nDims; i++)
	{
		mid[i] = (currentNode.minPoint[i] + currentNode.maxPoint[i]) / 2;
	}

	int ncount = 1;
	qrtcut[0] = queryrect;

	Point<T> pttmp(nDims);  //temporary point or corner
	for (int i = 0; i < nDims; i++)  //dimension iteration
	{
		int newadd = 0;
		for (int j = 0; j < ncount; j++)
		{
			if (qrtcut[j].minPoint[i] < mid[i] && qrtcut[j].maxPoint[i] > mid[i])
			{
				Rect<T> rtnew = qrtcut[j];
				pttmp = rtnew.minPoint;
				pttmp[i] = mid[i];
				rtnew.SetMinPoint(pttmp);

				pttmp = qrtcut[j].maxPoint;
				pttmp[i] = mid[i];
				qrtcut[j].SetMaxPoint(pttmp);

				qrtpos[ncount + newadd] = (1 << i) + qrtpos[j];
				qrtcut[ncount + newadd] = rtnew;

				newadd++;
			}

			if (qrtcut[j].minPoint[i] >= mid[i])  //all bigger than the middle line
			{
				qrtpos[j] |= 1 << i;  //just update its position---put 1 on the dimension bit
			}
		}  //end for rect count

		ncount += newadd;  //update all rectangle count
	}  //end for dimension

	for (int i = 0; i < ncount; i++)   //final rect number 
	{
		TreeNode<T> cNode = currentNode.GetChildNode(qrtpos[i]);
		int rec = cNode.Spatialrelationship(qrtcut[i]);
		if (rec == 0)
		{
			resultTNode.push_back(cNode);  //equal
		}
		else if (rec == -1)
		{
		}
		else
		{
			query_approximate(cNode, qrtcut[i], resultTNode);  //recursive query
		}
	}
}

///breadth-first traversal in the 2^n-ary tree
template<typename T>//, int nDims, int mBits
//void QueryBySFC<T, nDims, mBits>::query_approximate2(TreeNode<T, nDims> nd, Rect<T, nDims> queryrect, vector<TreeNode<T, nDims>>& resultTNode, int nranges, int ktimes)
void QueryBySFC<T>::query_approximate2(TreeNode<T> nd, Rect<T> queryrect, vector<TreeNode<T>>& resultTNode, int nranges, int ktimes)
{
	int nary_num = 1 << nDims;  //max count: 2^nDims

	typedef tuple<TreeNode<T>, Rect<T>> NRTuple;
	queue<NRTuple> query_queue;

	TreeNode<T> nchild;//=nd
	int res, last_level;
	///////////////////////////////////////////
	//queue the root node
	query_queue.push(NRTuple(nd, queryrect));
	last_level = 0;

	for (; !query_queue.empty(); query_queue.pop())
	{
		NRTuple currenttuple = query_queue.front();

		TreeNode<T> currentNode = std::get<0>(currenttuple);
		Rect<T> qrt = std::get<1>(currenttuple);

		//cout << currentNode.level << endl;
		//////////////////////////////////////////////////////
		//check the level and numbers of results
		if ((nranges != 0) && (last_level != currentNode.level) && (resultTNode.size() + query_queue.size() > ktimes * nranges)) //we are in the new level and full
		{
			///move all the left nodes in the queue to the resuts node vector
			for (; !query_queue.empty(); query_queue.pop())
			{
				resultTNode.push_back(std::get<0>(query_queue.front()));
			}

			break; //now
		}

		/////////////////////////////////////////////////////////////////////
		////get all children nodes till equal or intersect, if contain, continue to get children nodes
		do
		{
			for (int i = 0; i < nary_num; i++)
			{
				nchild = currentNode.GetChildNode(i);
				if (nchild.Spatialrelationship(qrt) == 0)  //equal: stop
				{
					resultTNode.push_back(nchild);
					res = 1;
					break; //break for and while ---to continue queue iteration
				}
				else if (nchild.Spatialrelationship(qrt) == 2)  //intersect: divide queryrectangle
				{
					res = 2;
					break;  //break for and while ---divide queryrectangle
				}
				else  if (nchild.Spatialrelationship(qrt) == 1)//contain: go down to the next level untill equal or intersect
				{
					res = 0;
					currentNode = nchild;
					break; //break for but to continue while
				}
			}//end for nary children
		} while (!res);

		if (res == 1) continue; //here break to continue for (queue iteration)
		//			
		//	//divide the input query rectangle into even parts, e.g. 2 or 4 parts
		//	//0~3 for 2d; upper 2|3----10|11;----- YX for 2D, ZYX for 3D, TZYX for 4D--each dim one bit
		//	//0~3 for 2d; lower 0|1----00|01 ------one dim: less = 0; greater = 1		
		vector<Rect<T>> qrtcut(nary_num);  //2^nDims parts
		vector<int> qrtpos(nary_num);  //the qrtcut corresponds to treenode

		for (int i = 0; i < nary_num; i++)
		{
			qrtpos[i] = 0;
		}

		vector<int> mid(nDims);  //middle cut line--dim number
		for (int i = 0; i < nDims; i++)
		{
			mid[i] = (currentNode.minPoint[i] + currentNode.maxPoint[i]) / 2;
		}

		int ncount = 1;
		qrtcut[0] = qrt;

		Point<T> pttmp(nDims);  //temporary point or corner
		for (int i = 0; i < nDims; i++)  //dimension iteration
		{
			int newadd = 0;
			for (int j = 0; j < ncount; j++)
			{
				if (qrtcut[j].minPoint[i] < mid[i] && qrtcut[j].maxPoint[i] > mid[i])
				{
					Rect<T> rtnew = qrtcut[j];
					pttmp = rtnew.minPoint;
					pttmp[i] = mid[i];
					rtnew.SetMinPoint(pttmp);

					pttmp = qrtcut[j].maxPoint;
					pttmp[i] = mid[i];
					qrtcut[j].SetMaxPoint(pttmp);

					qrtpos[ncount + newadd] = (1 << i) + qrtpos[j];
					qrtcut[ncount + newadd] = rtnew;

					newadd++;
				}

				if (qrtcut[j].minPoint[i] >= mid[i])  //all bigger than the middle line
				{
					qrtpos[j] |= 1 << i;  //just update its position---put 1 on the dimension bit
				}
			}//end for rect count

			ncount += newadd;  //update all rectangle count
		}//end for dimension

		for (int i = 0; i < ncount; i++)   //final rect number 
		{
			TreeNode<T> cNode = currentNode.GetChildNode(qrtpos[i]);
			int rec = cNode.Spatialrelationship(qrtcut[i]);
			if (rec == 0)
			{
				resultTNode.push_back(cNode);  //equal
			}
			else if (rec == -1)
			{
			}
			else
			{
				//query_approximate(cNode, qrtcut[i], resultTNode);  //recursive query
				query_queue.push(NRTuple(cNode, qrtcut[i]));
			}
		}//end for rect division check

	}///end for queue iteration
}
__global__ void recursive(sfc_bigint *d_a, sfc_bigint *d_b, sfc_bigint *d_c, sfc_bigint *d_d, sfc_bigint *d_z )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	sfc_bigint k = 0;//一开始忘记给z[]赋初值！！！一定是M！！！
	sfc_bigint row = d_a[id];
	sfc_bigint col = d_b[id];
	sfc_bigint row_2 = d_c[id];
	sfc_bigint col_2 = d_d[id];
	//printf("%d,%d\n", pre_row, pre_col);
	for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
		k |= (row & 1U << i) << (i + 1) | (col & 1U << i) << i;
		d_z[2*id] = k;
	}
	k = 0;
	for (int i = 0; i < sizeof(row) * CHAR_BIT; i++) {
		k |= (row_2 & 1U << i) << (i + 1) | (col_2 & 1U << i) << i;
		d_z[2*id+1] = k;
	}
}

template< typename T>//, int nDims, int mBits
//vector<sfc_bigint>  QueryBySFC<T, nDims, mBits>::RangeQueryByRecursive_LNG(Rect<T, nDims> queryrect, SFCType sfc_type, int nranges, int ktimes)
vector<sfc_bigint>  QueryBySFC<T>::RangeQueryByRecursive_LNG(Rect<T> queryrect, SFCType sfc_type, int nranges, int ktimes)
{
	//thrust::host_vector<TreeNode<T>> resultTNode;  //tree nodes correspond to queryRectangle
	vector<TreeNode<T>> resultTNode;
	TreeNode<T> root;  //root node
	root.level = 0;
	root.nDims = nDims;

	root.minPoint.nDims = root.maxPoint.nDims = nDims;
	for (int i = 0; i < nDims; i++)
	{
		root.minPoint[i] = 0;
		root.maxPoint[i] = 1 << mBits;

		queryrect.maxPoint[i] += 1;
	}

	int res = root.Spatialrelationship(queryrect);
	if (res == 0)  //equal
	{
		resultTNode.push_back(root);
	}
	if (res == 1)  //contain
	{
		query_approximate2(root, queryrect, resultTNode, nranges, ktimes);
	}
	
	
	//cout << resultTNode.size() << endl;

	int ncorners = 1 << nDims; //corner points number
	//vector<Point<T>> nodePoints(ncorners);
	//vector<sfc_bigint> node_vals(ncorners);

	map<sfc_bigint, sfc_bigint, less<sfc_bigint>> map_range;
	map<sfc_bigint, sfc_bigint, less<sfc_bigint>>::iterator itr;

	sfc_bigint val, r_start, r_end;
	//并行
	clock_t start_time = clock();
	sfc_bigint *a,*b,*c,*d,*z;
	a = (sfc_bigint*)malloc(sizeof(sfc_bigint)*resultTNode.size());
	b = (sfc_bigint*)malloc(sizeof(sfc_bigint)*resultTNode.size());
	c = (sfc_bigint*)malloc(sizeof(sfc_bigint)*resultTNode.size());
	d = (sfc_bigint*)malloc(sizeof(sfc_bigint)*resultTNode.size());
	z = (sfc_bigint*)malloc(2*sizeof(sfc_bigint)*resultTNode.size());
	for (int i = 0; i < resultTNode.size(); i++)
	{
		a[i]=resultTNode[i].minPoint[0];
		b[i]=resultTNode[i].minPoint[1];
		c[i]=resultTNode[i].maxPoint[0]-1;
		d[i]=resultTNode[i].maxPoint[1]-1;
	}
	sfc_bigint *d_a, *d_b, *d_c, *d_d,*d_z;
	cudaMalloc((void**)&d_a, resultTNode.size() * sizeof(sfc_bigint));
	cudaMalloc((void**)&d_b, resultTNode.size() * sizeof(sfc_bigint));
	cudaMalloc((void**)&d_c, resultTNode.size() * sizeof(sfc_bigint));
	cudaMalloc((void**)&d_d, resultTNode.size() * sizeof(sfc_bigint));
	cudaMalloc((void**)&d_z, 2* resultTNode.size() * sizeof(sfc_bigint));
	cudaMemcpy(d_a, a, resultTNode.size() * sizeof(sfc_bigint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, resultTNode.size() * sizeof(sfc_bigint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, resultTNode.size() * sizeof(sfc_bigint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, resultTNode.size() * sizeof(sfc_bigint), cudaMemcpyHostToDevice);
	int threadperblock = 1024;
	int blockpergrid = (resultTNode.size() + threadperblock - 1) / threadperblock;
	dim3 gridSize(blockpergrid);
	dim3 blockSize(threadperblock);
	recursive << <gridSize, blockSize>> >(d_a, d_b, d_c, d_d, d_z);
	cudaMemcpy((void*)z, (void*)d_z, 2 * sizeof(sfc_bigint)*resultTNode.size(), cudaMemcpyDeviceToHost);
	/*for (int i = 0; i < 2 * resultTNode.size(); i++)
	{
		cout << z[i] << endl;
	}*/
	//sfc_bigint k1, k2;
	//int num = resultTNode.size();//TreeNode<T>
	//thrust::host_vector<TreeNode<T>> vecLists;
	//vecLists = resultTNode;
	//thrust::device_vector<TreeNode<T>> d_vecLists;
	//d_vecLists = vecLists;
	//TreeNode<T>* vecListPtr;
	//cudaMalloc((void*)&vecListPtr);
	//vecListPtr = thrust::raw_pointer_cast(&d_vecLists[0]);
	/*int threadperblock = 64;
	int blockpergrid = (num+ threadperblock-1) / threadperblock;*/
	/*dim3 gridSize(blockpergrid);
	dim3 blockSize(threadperblock);*/
	//recursive << <gridSize, blockSize>>>>(vecListPtr);

	//cout << resultTNode[1].minPoint[1] << endl;

	//for (int i = 0; i < resultTNode.size(); i++)
	//{
	//	SFCConversion sfc(nDims, mBits);

	//	long long node_width = 1 << nDims * (mBits - resultTNode[i].level);//2^(n*(m-l))//每级结点的大小

	//	if (sfc_type == Hilbert) //encoding minPoint
	//	{
	//		val = sfc.HilbertEncode(resultTNode[i].minPoint);
	//	}
	//	if (sfc_type == Morton)
	//	{
	//		val = sfc.MortonEncode(resultTNode[i].minPoint);
	//	}

	//	r_start = val - val % node_width;
	//	r_end = r_start + node_width - 1;
	//	map_range[r_start] = r_end;
	//}

	/////////////////////////////////////////////////
	///merg continuous range--->if nranges=0, gap=1; if nranges !=0 ,find the Nth big gap
	///find the suitable distance dmin
	sfc_bigint dmin = 1;//for full ranges//合并阈值
	int nsize = map_range.size();
	if (nranges != 0) //not full ranges---control by nranges N
	{
		vector<sfc_bigint> vec_dist(nsize - 1);

		itr = map_range.begin();
		sfc_bigint last = itr->second;
		for (itr++; itr != map_range.end(); itr++)
		{
			vec_dist.push_back((itr->first - last));

			//cout << itr->first - last << endl;

			last = itr->second;
		}

		tbb::parallel_sort(vec_dist.begin(), vec_dist.end(), std::greater<sfc_bigint>());
		/*cout << "^^^^^^^^^^^^^^^^^^^^^^" << endl;
		thrust::sort(vec_dist.begin(), vec_dist.end());*/
		//for (int q = 0; q<nsize - 1;q++)
		//	cout << vec_dist[q]  << endl;

		dmin = vec_dist[nranges - 1];

		//cout << "min gap:" << dmin << endl;
	}

	//////merge
	sfc_bigint k1, k2;
	vector<sfc_bigint> rangevec;

	//itr = map_range.begin();
	//k1 = itr->first; //k1---k2 current range
	//k2 = itr->second;
	k1 = z[0];
	k2 = z[1];
	int i = 0;
	while (1)
	{
		//itr++; //get next range
		i = i + 2;
		//cout << k1  << ',' <<k2 << endl;
		if (i== 2*resultTNode.size())
		{
			rangevec.push_back(k1);
			rangevec.push_back(k2);

			break;
		}

		if ((z[i] - k2) <= dmin) // if the next range is continuous to k2 //itr->first == k2 + 1
		{
			//ncc++;
			k2 = z[i+1]; //enlarge current range
		}
		else //if the next range is not continuous to k2---sotre current range and start another search
		{
			rangevec.push_back(k1);
			rangevec.push_back(k2);

			k1 = z[i];
			k2 = z[i+1];
		}//end if
	}//end while	
	clock_t end_time = clock();
	float clockTime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;
	printf("Running time is: %3.2f ms\n", clockTime);
	//cout << rangevec.size();
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	cudaFree(d_z);
	free(a);
	free(b);
	free(c);
	free(d);
	free(z);
	return rangevec;
}
#endif
