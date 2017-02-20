/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__
#include "configure.h"

#if DEBUG
#include <iostream>
using namespace std;
#endif


namespace SDAI
{
/*
 * @note: define the Embedding layer
 * 	***This layer can only be used as the first layer in a model.
 * 	***The input data of this layer must be positive integer data type
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NB_SAMPLES, int INPUT_LENGTH  >
class Embedding
{
public:
	Embedding(const TYPE_T *WEIGHT)
	{
#if DEBUG
		cout<<"Embedding Layer......"<<endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
		cout<<"\tNB_SAMPLES = " << NB_SAMPLES << endl;
		cout<<"\tINPUT_LENGTH = " << INPUT_LENGTH << endl;
#endif
		/* initialize the weight */
		for( int i = 0; i < INPUT_DIM; i++)
		{
			for( int j = 0; j < OUTPUT_DIM; j++)
			{
				weight[i][j] = WEIGHT[i * OUTPUT_DIM + j];
			}
		}
	}

public:
	TYPE_T weight[INPUT_DIM][OUTPUT_DIM];
	TYPE_T res[NB_SAMPLES][INPUT_LENGTH][OUTPUT_DIM];

public:
	/*
	 * @note: the feedforward function
	 */
	void feedforward( TYPE_PINT data[NB_SAMPLES][INPUT_LENGTH] )
	{
		for( int i = 0; i < NB_SAMPLES; i++)
		{
#if EMBEDDING_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < INPUT_LENGTH; j++)
			{
#if EMBEDDING_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				TYPE_PINT index = data[i][j];
				for( int k = 0; k < OUTPUT_DIM; k++)
				{
#if EMBEDDING_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					res[i][j][k] = weight[index][k];
				}
			}
		}
	}
};


/*
 * @note: define a data stream based Embedding layer
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NB_SAMPLES, int INPUT_LENGTH  >
class Embedding_DataStream
{
public:
	Embedding_DataStream(const TYPE_T *WEIGHT)
	{
#if DEBUG
		cout<<"Embedding_DataStream Layer......"<<endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
		cout<<"\tNB_SAMPLES = " << NB_SAMPLES << endl;
		cout<<"\tINPUT_LENGTH = " << INPUT_LENGTH << endl;
#endif
		/* initialize the weight */
		for( int i = 0; i < INPUT_DIM; i++)
		{
			for( int j = 0; j < OUTPUT_DIM; j++)
			{
				weight[i][j] = WEIGHT[i * OUTPUT_DIM + j];
			}
		}
	}

public:
	TYPE_T weight[INPUT_DIM][OUTPUT_DIM];

public:
	/*
	 * @note: the feedforward function
	 */
	void feedforward(volatile TYPE_PINT *data, volatile TYPE_T *res)
	{
		for( int i = 0; i < NB_SAMPLES; i++)
		{
#if EMBEDDING_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < INPUT_LENGTH; j++)
			{
#if EMBEDDING_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				TYPE_PINT index = data[i * INPUT_LENGTH + j];
				for( int k = 0; k < OUTPUT_DIM; k++)
				{
#if EMBEDDING_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					res[ i * INPUT_LENGTH * OUTPUT_DIM + j * OUTPUT_DIM + k] = weight[index][k];
				}
			}
		}
	}
};


}

#endif
