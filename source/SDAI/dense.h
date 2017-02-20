/*
 * @author: jjf, Fudan University
 * @date: 2016/10/8
 */
#ifndef __DENSE__H__
#define __DENSE__H__

#include "activation.h"
#include "configure.h"
#include "mem.h"
#include <assert.h>
#include <string.h>

#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{
/*
 * @note: the Fully Connected Layer: Dense Layer
 * 	the input_shape = {INPUT_DIM}
 * 	the output shape = {OUTPUT_DIM}
 */
template<int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN>
class Dense
{
public:
	Dense(const TYPE_T *WEIGHT)
	{
		assert(INPUT_DIM > 0);
		assert(OUTPUT_DIM > 0);
#if DEBUG
		cout<<"Dense Layer......"<<endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
		/* initialize the weight */
		for( int i = 0; i < INPUT_DIM + 1; i++)
		{
			for( int j = 0; j < OUTPUT_DIM; j++)
				weight[i][j] = WEIGHT[i*OUTPUT_DIM + j];
		}
	}
public:
	TYPE_T	weight[INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	res[OUTPUT_DIM];

public:
	/*
	 * @note: the feedforword function
	 * @params: the input data is a 1D array with INPUT_DIM
	 */
	void feedforward(TYPE_T data[INPUT_DIM])
	{
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if DENSE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

			/* calculate the weight and bias*/
			TYPE_T tmp = weight[INPUT_DIM][i];
			for(int j = 0; j < INPUT_DIM; j++)
			{
#if DENSE_PERF_MODE == PERF_LOW || DENSE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				tmp += data[j] * weight[j][i];
			}

			/* calculate the activation function */
			res[i] = activation_fn<AC_FN>(tmp);
		}

		/* for the activation of softmax */
		if( AC_FN == SOFTMAX )
		{
			activation_softmax<OUTPUT_DIM>(res);
		}
	}
};

/*
 * @note: the Fully Connected Layer with streamed weight
 * 	the input_shape = {INPUT_DIM}
 * 	the output shape = {OUTPUT_DIM}
 */
template<int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN>
class Dense_WeightStream
{
public:
	Dense_WeightStream()
	{
		assert(INPUT_DIM > 0);
		assert(OUTPUT_DIM > 0);
#if DEBUG
		cout<<"StreamDense Layer......"<<endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	}
public:
	TYPE_T	res[OUTPUT_DIM];

public:
	/*
	 * @note: the feedforword function
	 * @params: the input data is a 1D array with INPUT_DIM
	 */
	void feedforward(volatile TYPE_T *weight, TYPE_T data[INPUT_DIM])
	{
		/* define a 1D line buffer */
		LineBuffer1D<INPUT_DIM + 1>		buffer;

		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if DENSE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			/* copy the weight from the M_AXI to local ram */
			buffer.fill(&weight[i * (INPUT_DIM + 1)]);

			/* calculate the weight and bias*/
			TYPE_T tmp = buffer.getval( INPUT_DIM );

			for(int j = 0; j < INPUT_DIM; j++)
			{
#if DENSE_PERF_MODE == PERF_LOW || DENSE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				tmp += data[j] * buffer.getval(j);
			}

			/* calculate the activation function */
			res[i] = activation_fn<AC_FN>(tmp);
		}

		/* for the activation of softmax */
		if( AC_FN == SOFTMAX )
		{
			activation_softmax<OUTPUT_DIM>(res);
		}
	}
};

}


#endif
