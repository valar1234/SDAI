/*
 * @author: jjf, Fudan University
 * @date: 2016/10/10
 */
#ifndef __CONVOLUTION1D__
#define __CONVOLUTION1D__
#include "activation.h"
#include <assert.h>
#include "reshape.h"
#include "mem.h"
#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

#define CONV1D_OUTPUT_DIM(STEP, FILTER_LENGTH, SUBSAMPLE_LENGTH)		((STEP - FILTER_LENGTH)/SUBSAMPLE_LENGTH + 1)
/*
 * @note: define the 1-dimension convolution layer
 * 			the input_shape = {STEP * INPUT_DIM}
 * 			the output_shape = {OUTPUT_DIM * INPUT_DIM}
 * @params:
 * 		weight is a 3-D array(FILTER_LENGTH x INPUT_DIM x NB_FILTER) and then expand as 1-D array.
 * 		bias is a 1-D array, and its length is NB_FILTER
 * 		res is the layer output
 */
template<int NB_FILTER, int FILTER_LENGTH, int STEP, int INPUT_DIM = 1, int SUBSAMPLE_LENGTH = 1, ACTIVATION AC_FN = LINEAR, int OUTPUT_DIM=((STEP - FILTER_LENGTH)/SUBSAMPLE_LENGTH + 1)>
class Convolution1D
{
public:
	Convolution1D(const TYPE_T *WEIGHT, const TYPE_T *BIAS)
	{
		assert(STEP > FILTER_LENGTH);
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete

#if DEBUG
		cout<<"Convolution1D Layer......"<<endl;
		cout<<"\tNB_FILTER = " << NB_FILTER << endl;
		cout<<"\tFILTER_LENGTH = " << FILTER_LENGTH << endl;
		cout<<"\tSTEP = " << STEP << endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tSUBSAMPLE_LENGTH = " << SUBSAMPLE_LENGTH << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
		/* initialize the weight and bias */
		for( int i = 0; i < FILTER_LENGTH; i++)
		{
			for( int j = 0; j < INPUT_DIM; j++)
			{
				for( int k = 0; k < NB_FILTER; k++)
				{
					weight[i][j][k] = WEIGHT[i*INPUT_DIM*NB_FILTER + j*NB_FILTER + k];
				}
			}
		}
		for(int k = 0; k < NB_FILTER; k++)
		{
			bias[k] = BIAS[k];
		}
	}

public:
	TYPE_T	weight[FILTER_LENGTH][INPUT_DIM][NB_FILTER];
	TYPE_T	bias[NB_FILTER];
	TYPE_T	res[OUTPUT_DIM][NB_FILTER];

public:
	/*
	 * @note: define the feedforward function
	 * @params: data is a STEP * INPUT_DIM 2D array
	 */
	void feedforward(TYPE_T data[STEP][INPUT_DIM])
	{
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if CONVOLUTION1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < NB_FILTER; j++)
			{
#if CONVOLUTION1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the weight and bias */
				TYPE_T t = bias[j];

				for( int k = 0; k < FILTER_LENGTH; k++)
				{
					for( int v  = 0; v < INPUT_DIM ; v++)
					{
#if CONVOLUTION1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
						t += data[i*SUBSAMPLE_LENGTH + k][v] * weight[k][v][j];
					}
				}


				/* calculate the activation function */
				res[i][j] = activation_fn<AC_FN>(t);
			}
		}
	}
};

/*
 * @note: define the data stream based convolution1D
 */
template<int NB_FILTER, int FILTER_LENGTH, int STEP, int INPUT_DIM = 1, int SUBSAMPLE_LENGTH = 1, ACTIVATION AC_FN = LINEAR, int OUTPUT_DIM=((STEP - FILTER_LENGTH)/SUBSAMPLE_LENGTH + 1)>
class Convolution1D_DataStream
{
public:
	Convolution1D_DataStream(const TYPE_T *WEIGHT, const TYPE_T *BIAS)
	{
		assert(STEP > FILTER_LENGTH);
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete

#if DEBUG
		cout<<"Convolution1D_DataStream Layer......"<<endl;
		cout<<"\tNB_FILTER = " << NB_FILTER << endl;
		cout<<"\tFILTER_LENGTH = " << FILTER_LENGTH << endl;
		cout<<"\tSTEP = " << STEP << endl;
		cout<<"\tINPUT_DIM = " << INPUT_DIM << endl;
		cout<<"\tSUBSAMPLE_LENGTH = " << SUBSAMPLE_LENGTH << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
		/* initialize the weight and bias */
		for( int i = 0; i < FILTER_LENGTH; i++)
		{
			for( int j = 0; j < INPUT_DIM; j++)
			{
				for( int k = 0; k < NB_FILTER; k++)
				{
					weight[i][j][k] = WEIGHT[i*INPUT_DIM*NB_FILTER + j*NB_FILTER + k];
				}
			}
		}
		for(int k = 0; k < NB_FILTER; k++)
		{
			bias[k] = BIAS[k];
		}
	}

public:
	TYPE_T	weight[FILTER_LENGTH][INPUT_DIM][NB_FILTER];
	TYPE_T	bias[NB_FILTER];

public:
	/*
	 * @note: define the feedforward function
	 * @params: data is a STEP * INPUT_DIM 2D array
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res)
	{
#if CONVOLUTION1D_OPT_MODE == OPT_BUFFER
		/* define the line buffer and window buffer */
		LineBuffer2D<FILTER_LENGTH, INPUT_DIM, SUBSAMPLE_LENGTH>				l_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );

#elif CONVOLUTION1D_OPT_MODE == OPT_MEM
		/* define a local BRAM*/
		Reshape_Stream_2D<FILTER_LENGTH, INPUT_DIM>			stream;
#endif
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if CONVOLUTION1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if CONVOLUTION1D_OPT_MODE == OPT_BUFFER
			if( i > 0 && i < OUTPUT_DIM)
			{
				l_buffer.shift_up();
				l_buffer.fill_line(&data[(i * SUBSAMPLE_LENGTH + FILTER_LENGTH - 1) * INPUT_DIM] );
			}

#elif CONVOLUTION1D_OPT_MODE == OPT_MEM
			/* copy data from AXI master to local BRAM */
			stream.feedforward( &data[(i * SUBSAMPLE_LENGTH ) * INPUT_DIM ]);
#endif

			for( int j = 0; j < NB_FILTER; j++)
			{
#if CONVOLUTION1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the weight and bias */
				TYPE_T t = bias[j];

				for( int k = 0; k < FILTER_LENGTH; k++)
				{
					for( int v  = 0; v < INPUT_DIM ; v++)
					{
#if CONVOLUTION1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if CONVOLUTION1D_OPT_MODE == OPT_BUFFER
						TYPE_T val = l_buffer.getval(k, v);
#elif CONVOLUTION1D_OPT_MODE == OPT_MEM
						TYPE_T val = stream.res[k][v];
#else
						TYPE_T val = data[(i * SUBSAMPLE_LENGTH + k) * INPUT_DIM + v ];
#endif
						t += val * weight[k][v][j];
					}
				}


				/* calculate the activation function */
				res[i * NB_FILTER + j] = activation_fn<AC_FN>(t);
			}
		}
	}
};

}


#endif
