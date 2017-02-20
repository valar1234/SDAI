/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __POOLING2D_H__
#define __POOLING2D_H__
#include "configure.h"
#include <assert.h>
#include "reshape.h"

#if 1
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

/*
 * @note: 2D Maximum Pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW = 2, int POOL_COL = 2, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class MaxPooling2D
{
public:
	MaxPooling2D()
	{
		assert(OUT_ROW > 0);
		assert(OUT_COL > 0);
#if DEBUG
		cout<<"MaxPooling2D Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:
	TYPE_T res[OUT_ROW][OUT_COL][NB];

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
	void feedforward(TYPE_T data[ROW][COL][NB])
	{
		MAXPOOLING2D: for (int row = 0; row < OUT_ROW; row++)
		{
			for (int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the maximum value in the local window*/
					TYPE_T max = data[row * POOL_ROW][col * POOL_COL][k];
					for (int i = 0; i < POOL_ROW; i++)
					{
						for (int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
							TYPE_T v = data[row * POOL_ROW + i][col * POOL_COL + j][k];
							if (v > max)
								max = v;
						}
					}
					res[row][col][k] = max;
				}
			}
		}
	}
};

/*
 * @note: 2D Maximum Pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW = 2, int POOL_COL = 2, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class MaxPooling2D_Stream
{
public:
	MaxPooling2D_Stream()
	{
		assert(OUT_ROW > 0);
		assert(OUT_COL > 0);
#if DEBUG
		cout<<"MaxPooling2D_Stream Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res)
	{

#if POOLING2D_OPT_MODE == OPT_BUFFER
		/* define a 3D LineBuffer */
		LineBuffer3D<POOL_ROW, COL, NB, POOL_ROW>								l_buffer;
		WindowBuffer3D<POOL_ROW, POOL_COL, NB, COL, POOL_ROW, POOL_COL>			w_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );
		w_buffer.fill( l_buffer, 0 );

#elif POOLING2D_OPT_MODE == OPT_MEM
		/* define a local BRAM to store part of the data */
		Reshape_Stream_3D<POOL_ROW, COL, NB>	stream;

#endif

		MAXPOOLING2D: for (int row = 0; row < OUT_ROW; row++)
		{
#if POOLING2D_OPT_MODE == OPT_BUFFER
			/* update the 3D LineBuffer*/
			if( row > 0 && row < OUT_ROW)
			{
				l_buffer.shift_up();
				l_buffer.fill_line( &data[(row  * POOL_ROW ) * COL * NB] );
				w_buffer.fill( l_buffer, 0 );
			}

#elif POOLING2D_OPT_MODE == OPT_MEM
			/*copy data from AXI master to local BRAM */
			stream.feedforward(&data[(row * POOL_ROW ) * COL * NB]);

#endif

			for (int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if POOLING2D_OPT_MODE == OPT_BUFFER
				/* update the window buffer */
				if( col > 0 && col < OUT_COL)
				{
					w_buffer.shift_left();
					w_buffer.insert_right(l_buffer, col * POOL_COL );
				}
#endif

				for (int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the maximum value in the local window*/

#if POOLING2D_OPT_MODE == OPT_BUFFER
					TYPE_T max = w_buffer.getval(0, 0, k);
#elif POOLING2D_OPT_MODE == OPT_MEM
					TYPE_T max = stream.res[0][col * POOL_COL][k];
#else
					TYPE_T max = data[(row * POOL_ROW ) * COL * NB + (col * POOL_COL) * NB + k];
#endif
					for (int i = 0; i < POOL_ROW; i++)
					{
						for (int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if POOLING2D_OPT_MODE == OPT_BUFFER
							TYPE_T v = w_buffer.getval(i, j, k);
#elif POOLING2D_OPT_MODE == OPT_MEM
							TYPE_T v = stream.res[i][col * POOL_COL + j][k];
#else
							TYPE_T v = data[(row * POOL_ROW + i) * COL * NB + (col * POOL_COL + j) * NB + k];
#endif
							if (v > max)
								max = v;
						}
					}
					res[row * OUT_COL * NB + col * NB + k] = max;
				}
			}
		}
	}
};


/*
 * @note: the 2D average pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW, int POOL_COL, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class AveragePooling2D
{
public:
	AveragePooling2D()
	{
#if DEBUG
		cout<<"AveragePooling2D Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:
	TYPE_T res[OUT_ROW][OUT_COL][NB];

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
	void feedforward(TYPE_T data[ROW][COL][NB])
	{
		for( int row = 0; row < OUT_ROW; row++)
		{
			for( int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for( int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the mean value in the local window*/
					TYPE_T sum = 0;
					for( int i = 0; i < POOL_ROW; i++)
					{
						for( int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
							sum += data[row*POOL_ROW + i][col*POOL_COL + j][k];
						}
					}
					res[row][col][k] = sum/(POOL_ROW * POOL_COL);
				}
			}
		}
	}
};


/*
 * @note: the 2D average pooling layer, normally used after Convolution2D layer
 */
template<int ROW, int COL, int NB, int POOL_ROW, int POOL_COL, int OUT_ROW = ROW/POOL_ROW, int OUT_COL = COL/POOL_COL>
class AveragePooling2D_Stream
{
public:
	AveragePooling2D_Stream()
	{
#if DEBUG
		cout<<"AveragePooling2D_Stream Layer......"<<endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tNB = " << NB << endl;
		cout<<"\tPOOL_ROW = " << POOL_ROW << endl;
		cout<<"\tPOOL_COL = " << POOL_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif
	}
public:
	TYPE_T res[OUT_ROW][OUT_COL][NB];

public:
	/*
	 * @note: the input data is ROW x COL x NB 3D array
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res )
	{
#if POOLING2D_OPT_MODE == OPT_BUFFER
		/* define a 3D LineBuffer */
		LineBuffer3D<POOL_ROW, COL, NB, POOL_ROW>								l_buffer;
		WindowBuffer3D<POOL_ROW, POOL_COL, NB, COL, POOL_ROW, POOL_COL>			w_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );
		w_buffer.fill( l_buffer, 0 );

#elif POOLING2D_OPT_MODE == OPT_MEM
		/* define a local BRAM to store part of the data */
		Reshape_Stream_3D<POOL_ROW, COL, NB>	stream;

#endif

		for( int row = 0; row < OUT_ROW; row++)
		{
#if POOLING2D_OPT_MODE == OPT_BUFFER
			/* update the 3D LineBuffer*/
			if( row > 0 && row < OUT_ROW)
			{
				l_buffer.shift_up();
				l_buffer.fill_line( &data[(row  * POOL_ROW ) * COL * NB] );
				w_buffer.fill( l_buffer, 0 );
			}

#elif POOLING2D_OPT_MODE == OPT_MEM
			/*copy data from AXI master to local BRAM */
			stream.feedforward(&data[(row * POOL_ROW ) * COL * NB]);

#endif

			for( int col = 0; col < OUT_COL; col++)
			{
#if POOLING2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if POOLING2D_OPT_MODE == OPT_BUFFER
				/* update the window buffer */
				if( col > 0 && col < OUT_COL)
				{
					w_buffer.shift_left();
					w_buffer.insert_right(l_buffer, col * POOL_COL );
				}
#endif

				for( int k = 0; k < NB; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if POOLING2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the mean value in the local window*/
					TYPE_T sum = 0;
					for( int i = 0; i < POOL_ROW; i++)
					{
						for( int j = 0; j < POOL_COL; j++)
						{
#if POOLING2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if POOLING2D_OPT_MODE == OPT_BUFFER
							TYPE_T v = w_buffer.getval(i, j, k);
#elif POOLING2D_OPT_MODE == OPT_MEM
							TYPE_T v = stream.res[i][col * POOL_COL + j][k];
#else
							TYPE_T v = data[(row * POOL_ROW + i) * COL * NB + (col * POOL_COL + j) * NB + k];
#endif
							sum += v;

						}
					}
					res[row * OUT_COL * NB + col * NB + k] = sum/(POOL_ROW * POOL_COL);
				}
			}
		}
	}
};


}


#endif
