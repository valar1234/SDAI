/*
 * @author: jjf, Fudan University
 * @date: 2016/10/10
 */
#ifndef __POOLING_H__
#define __POOLING_H__
#include <assert.h>
#include "configure.h"

#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

#define POOLING1D_OUTPUT_DIM(DIM1, POOL_LENGTH)		(DIM1/POOL_LENGTH)

/*
 * @note: define the maxpool1D layer
 * @params: DIM1 is the first Dimension size for the input shape
 * 			DIM2 is the second Dimension size for the input shape
 * 			OUTPUT_DIM is the first Dimension size for the output shape
 */
template<int POOL_LENGTH, int DIM1, int DIM2, int OUTPUT_DIM = (DIM1/POOL_LENGTH)>
class MaxPooling1D
{
public:
	MaxPooling1D()
	{
		assert(OUTPUT_DIM > 0 );
#if DEBUG
		cout<<"MaxPooling1D Layer......"<<endl;
		cout<<"\tDIM1 = " << DIM1 << endl;
		cout<<"\tDIM2 = " << DIM2 << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};
	TYPE_T	res[OUTPUT_DIM][DIM2];

public:
	/*
	 * @note: feedforward function
	 */
	void feedforward(TYPE_T data[DIM1][DIM2])
	{
		TYPE_T max;
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if POOLING1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if POOLING1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the maximum value in the POOL_LENGTH */
				max = data[i * POOL_LENGTH][j];
				for(int k = 1; k < POOL_LENGTH; k++)
				{
#if POOLING1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					TYPE_T tmp = data[i * POOL_LENGTH + k][j];
					if( tmp > max)
						max = tmp;
				}
				res[i][j] = max;
			}
		}
	}

};

/*
 * @note:  stream-based MaxPooling1D
 */
template<int POOL_LENGTH, int DIM1, int DIM2, int OUTPUT_DIM = (DIM1/POOL_LENGTH)>
class MaxPooling1D_Stream
{
public:
	MaxPooling1D_Stream()
	{
		assert(OUTPUT_DIM > 0 );
#if DEBUG
		cout<<"MaxPooling1D_Stream Layer......"<<endl;
		cout<<"\tDIM1 = " << DIM1 << endl;
		cout<<"\tDIM2 = " << DIM2 << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
	/*
	 * @note: feedforward function
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res)
	{
#if POOLING1D_OPT_MODE == OPT_BUFFER
		/* define the line buffer and window buffer */
		LineBuffer2D<POOL_LENGTH, DIM2, POOL_LENGTH>				l_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );

#elif POOLING1D_OPT_MODE == OPT_MEM
		/* define a local BRAM*/
		Reshape_Stream_2D<POOL_LENGTH, DIM2>						stream;
#pragma HLS ARRAY_PARTITION variable=stream.res dim=1 complete
#endif
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if POOLING1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if POOLING1D_OPT_MODE == OPT_BUFFER
			if( i > 0 && i < OUTPUT_DIM)
			{
				l_buffer.shift_up();
				l_buffer.fill_line( &data[(i * POOL_LENGTH ) * DIM2] );
			}

#elif POOLING1D_OPT_MODE == OPT_MEM
			/* copy data from AXI master to local BRAM */
			stream.feedforward( &data[(i * POOL_LENGTH) * DIM2 ]);
#endif

			for( int j = 0; j < DIM2; j++)
			{
#if POOLING1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif

				/* calculate the maximum value in the POOL_LENGTH */
#if POOLING1D_OPT_MODE == OPT_BUFFER
				TYPE_T max = l_buffer.getval(0, j);
#elif POOLING1D_OPT_MODE == OPT_MEM
				TYPE_T max = stream.res[0][j];

#else
				TYPE_T max = data[(i * POOL_LENGTH) * DIM2 + j];
#endif

				for(int k = 1; k < POOL_LENGTH; k++)
				{
#if POOLING1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if POOLING1D_OPT_MODE == OPT_BUFFER
					TYPE_T val = l_buffer.getval(k, j);
#elif POOLING1D_OPT_MODE == OPT_MEM
					TYPE_T val = stream.res[k][j];
#else
					TYPE_T val = data[(i * POOL_LENGTH + k) * DIM2 + j];
#endif
					if( val > max)
						max = val;
				}
				/* save the result */
				res[i * DIM2 + j] = max;
			}
		}
	}

};

/*
 * @note: define the Averagepooling1D layer
 * @params: DIM1 is the first Dimension size for the input shape
 * 			DIM2 is the second Dimension size for the input shape
 * 			OUTPUT_DIM is the first Dimension size for the output shape
 */
template<int POOL_LENGTH, int DIM1, int DIM2, int OUTPUT_DIM = (DIM1/POOL_LENGTH)>
class AveragePooling1D
{
public:
	AveragePooling1D()
	{
#if DEBUG
		cout<<"AveragePooling1D Layer......"<<endl;
		cout<<"\tDIM1 = " << DIM1 << endl;
		cout<<"\tDIM2 = " << DIM2 << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};
	TYPE_T res[OUTPUT_DIM][DIM2];

public:
	void feedforward(TYPE_T data[DIM1][DIM2])
	{
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if POOLING1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if POOLING1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the average value in the POOL_LENGTH */
				TYPE_T sum = 0;
				for( int k = 0; k < POOL_LENGTH; k++)
				{
#if POOLING1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					sum += data[i * POOL_LENGTH + k ][j];
				}

				res[i][j] = sum/POOL_LENGTH;
			}
		}
	}

};

template<int POOL_LENGTH, int DIM1, int DIM2, int OUTPUT_DIM = (DIM1/POOL_LENGTH)>
class AveragePooling1D_Stream
{
public:
	AveragePooling1D_Stream()
	{
#if DEBUG
		cout<<"AveragePooling1D_Stream Layer......"<<endl;
		cout<<"\tDIM1 = " << DIM1 << endl;
		cout<<"\tDIM2 = " << DIM2 << endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
	/*
	 * @note: the feed forward function
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res)
	{
#if POOLING1D_OPT_MODE == OPT_BUFFER
		/* define the line buffer and window buffer */
		LineBuffer2D<POOL_LENGTH, DIM2, POOL_LENGTH>				l_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );

#elif POOLING1D_OPT_MODE == OPT_MEM
		/* define a local BRAM*/
		Reshape_Stream_2D<POOL_LENGTH, DIM2>						stream;
#endif

		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#if POOLING1D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if POOLING1D_OPT_MODE == OPT_BUFFER
			if( i > 0 && i < OUTPUT_DIM)
			{
				l_buffer.shift_up();
				l_buffer.fill_line( &data[(i * POOL_LENGTH ) * DIM2] );
			}

#elif POOLING1D_OPT_MODE == OPT_MEM
			/* copy data from AXI master to local BRAM */
			stream.feedforward( &data[(i * POOL_LENGTH) * DIM2 ]);
#endif

			for( int j = 0; j < DIM2; j++)
			{
#if POOLING1D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the average value in the POOL_LENGTH */
				TYPE_T sum = 0;
				for( int k = 0; k < POOL_LENGTH; k++)
				{
#if POOLING1D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if POOLING1D_OPT_MODE == OPT_BUFFER
					TYPE_T val = l_buffer.getval(k, j);
#elif POOLING1D_OPT_MODE == OPT_MEM
					TYPE_T val = stream.res[k][j];
#else
					TYPE_T val = data[(i * POOL_LENGTH + k) * DIM2 + j];
#endif
					sum += val;
				}
				/* save the result */
				res[i * DIM2 + j] = sum/POOL_LENGTH;
			}
		}
	}

};

}

#endif
