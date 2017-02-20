/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __CONVOLUTION2D_H__
#define __CONVOLUTION2D_H__
#include "activation.h"
#include "configure.h"
#include <assert.h>
#include "mem.h"
#include "reshape.h"
#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

/*
 * @note: define the convolution2D layer
 */

template<int NB_FILTER, int NB_ROW, int NB_COL, int ROW, int COL, int INPUT_DIM = 1, ACTIVATION AC_FN=LINEAR, int SUBSAMPLE_ROW=1, int SUBSAMPLE_COL=1,
		int OUT_ROW=(ROW - NB_ROW)/SUBSAMPLE_ROW + 1, int OUT_COL=(COL - NB_COL)/SUBSAMPLE_COL + 1 >
class Convolution2D
{
public:
	Convolution2D(const TYPE_T *WEIGHT, const TYPE_T *BIAS)
	{
		assert(ROW > NB_ROW);
		assert(COL > NB_COL);
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#if DEBUG
		cout<<"Convolution2D Layer......"<<endl;
		cout<<"\tNB_FILTER = " << NB_FILTER << endl;
		cout<<"\tNB_ROW = " << NB_ROW << endl;
		cout<<"\tNB_COL = " << NB_COL << endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tINPUT_DIM = "<<INPUT_DIM << endl;
		cout<<"\tSUBSAMPLE_ROW = " << SUBSAMPLE_ROW << endl;
		cout<<"\tSUBSAMPLE_COL = " << SUBSAMPLE_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif

		/* initialize the weight and bias */
		for( int i = 0; i < NB_ROW; i++)
		{
			for( int j = 0; j < NB_COL; j++)
			{
				for( int m = 0; m < INPUT_DIM; m++)
				{
					for(int n = 0; n < NB_FILTER; n++)
						weight[i][j][m][n] = WEIGHT[ i*NB_COL*INPUT_DIM*NB_FILTER + j*INPUT_DIM*NB_FILTER + m*NB_FILTER + n];
				}
			}
		}
		for( int i = 0; i < NB_FILTER; i++)
		{
			bias[i] = BIAS[i];
		}
	}
public:
	/*the weights is a 4D array with NB_ROW * NB_COL * INPUT_DIM * NB_FILTER */
	TYPE_T	weight[NB_ROW][NB_COL][INPUT_DIM][NB_FILTER];
	/*the bias is a 1D array with NB_FILTER */
	TYPE_T	bias[NB_FILTER];
	TYPE_T res[OUT_ROW][OUT_COL][NB_FILTER];

public:
	/*
	 * @note: the feedback function
	 * @params: the input data is a 3D array, ROW * COL * INPUT_DIM
	 */
	void feedforward(TYPE_T data[ROW][COL][INPUT_DIM])
	{
		for( int row = 0; row < OUT_ROW; row++)
		{
			for( int col = 0; col < OUT_COL; col++)
			{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
				for (int k = 0; k < NB_FILTER; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the weight and bias */
					TYPE_T t = bias[k];

					for (int m = 0; m < NB_ROW; m++)
					{
						for (int n = 0; n < NB_COL; n++)
						{
							for (int v = 0; v < INPUT_DIM; v++)
							{
#if CONVOLUTION2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
								t += data[row * SUBSAMPLE_ROW + m][col * SUBSAMPLE_COL + n][v] * weight[m][n][v][k];
							}

						}
					}

					/* calculate the activation function */
					res[row][col][k] = activation_fn<AC_FN>(t);
				}
			}
		}
	}
};



/*
 * @note: define the Convolution2D_DataStream layer
 */

template<int NB_FILTER, int NB_ROW, int NB_COL, int ROW, int COL, int INPUT_DIM = 1, ACTIVATION AC_FN=LINEAR, int SUBSAMPLE_ROW=1, int SUBSAMPLE_COL=1,
		int OUT_ROW=(ROW - NB_ROW)/SUBSAMPLE_ROW + 1, int OUT_COL=(COL - NB_COL)/SUBSAMPLE_COL + 1 >
class Convolution2D_DataStream
{
public:
	Convolution2D_DataStream(const TYPE_T *WEIGHT, const TYPE_T *BIAS)
	{
		assert(ROW > NB_ROW);
		assert(COL > NB_COL);
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete

#if DEBUG
		cout<<"Convolution2D_DataStream Layer......"<<endl;
		cout<<"\tNB_FILTER = " << NB_FILTER << endl;
		cout<<"\tNB_ROW = " << NB_ROW << endl;
		cout<<"\tNB_COL = " << NB_COL << endl;
		cout<<"\tROW = " << ROW << endl;
		cout<<"\tCOL = " << COL << endl;
		cout<<"\tINPUT_DIM = "<<INPUT_DIM << endl;
		cout<<"\tSUBSAMPLE_ROW = " << SUBSAMPLE_ROW << endl;
		cout<<"\tSUBSAMPLE_COL = " << SUBSAMPLE_COL << endl;
		cout<<"\tOUT_ROW = " << OUT_ROW << endl;
		cout<<"\tOUT_COL = " << OUT_COL << endl;
#endif

		/* initialize the weight and bias */
		for( int i = 0; i < NB_ROW; i++)
		{
			for( int j = 0; j < NB_COL; j++)
			{
				for( int m = 0; m < INPUT_DIM; m++)
				{
					for(int n = 0; n < NB_FILTER; n++)
						weight[i][j][m][n] = WEIGHT[ i*NB_COL*INPUT_DIM*NB_FILTER + j*INPUT_DIM*NB_FILTER + m*NB_FILTER + n];
				}
			}
		}
		for( int i = 0; i < NB_FILTER; i++)
		{
			bias[i] = BIAS[i];
		}
	}
public:
	/*the weights is a 4D array with NB_ROW * NB_COL * INPUT_DIM * NB_FILTER */
	TYPE_T	weight[NB_ROW][NB_COL][INPUT_DIM][NB_FILTER];
	/*the bias is a 1D array with NB_FILTER */
	TYPE_T	bias[NB_FILTER];

public:


	/*
	 * @note: the feedback function
	 * @params: the input data is a 3D array, ROW * COL * INPUT_DIM
	 */
	void feedforward(volatile TYPE_T *data, volatile TYPE_T *res)
	{

#if CONVOLUTION2D_OPT_MODE == OPT_BUFFER
		/* define a 3D LineBuffer */
		LineBuffer3D<NB_ROW, COL, INPUT_DIM, SUBSAMPLE_ROW>									l_buffer;
		WindowBuffer3D<NB_ROW, NB_COL, INPUT_DIM, COL, SUBSAMPLE_ROW, SUBSAMPLE_COL>		w_buffer;

		/* fill the line buffer and window buffer */
		l_buffer.fill( data );
		w_buffer.fill( l_buffer, 0 );

#elif CONVOLUTION2D_OPT_MODE == OPT_MEM
		/* define a local BRAM to store part of the data */
		Reshape_Stream_3D<NB_ROW, COL, INPUT_DIM>											stream;
#pragma HLS ARRAY_PARTITION variable=stream.res dim=1 complete

#endif

		for( int row = 0; row < OUT_ROW; row++)
		{
#if CONVOLUTION2D_OPT_MODE == OPT_BUFFER
			/* update the 3D LineBuffer*/
			if( row > 0 && row < OUT_ROW)
			{
				l_buffer.shift_up();
				l_buffer.fill_line( &data[((row + NB_ROW - 1) * SUBSAMPLE_ROW ) * COL * INPUT_DIM] );
				w_buffer.fill( l_buffer, 0 );
			}

#elif CONVOLUTION2D_OPT_MODE == OPT_MEM
			/*copy data from AXI master to local BRAM */
			stream.feedforward(&data[(row * SUBSAMPLE_ROW ) * COL * INPUT_DIM]);

#endif

			/* process the 2D convolution */
			for( int col = 0; col < OUT_COL; col++)
			{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif

#if CONVOLUTION2D_OPT_MODE == OPT_BUFFER
				/* update the window buffer */
				if( col > 0 && col < OUT_COL)
				{
					w_buffer.shift_left();
					w_buffer.insert_right(l_buffer, (col + NB_COL - 1) * SUBSAMPLE_COL);
				}
#endif

				for (int k = 0; k < NB_FILTER; k++)
				{
#if CONVOLUTION2D_PERF_MODE == PERF_HIGH || CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS LOOP_FLATTEN
#endif
#if CONVOLUTION2D_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
					/* calculate the weight and bias */
					TYPE_T t = bias[k];

					for (int m = 0; m < NB_ROW; m++)
					{
						for (int n = 0; n < NB_COL; n++)
						{
							for (int v = 0; v < INPUT_DIM; v++)
							{
#if CONVOLUTION2D_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif

#if CONVOLUTION2D_OPT_MODE == OPT_BUFFER
								TYPE_T val = w_buffer.getval(m, n, v);
#elif CONVOLUTION2D_OPT_MODE == OPT_MEM
								TYPE_T val = stream.res[m][col * SUBSAMPLE_COL + n][v];
#else
								TYPE_T val = data[(row * SUBSAMPLE_ROW + m) * COL * INPUT_DIM + (col * SUBSAMPLE_COL + n) * INPUT_DIM + v];
#endif
								t += val * weight[m][n][v][k];
							}

						}
					}

					/* calculate the activation function */
					res[row * OUT_COL * NB_FILTER + col * NB_FILTER + k] = activation_fn<AC_FN>(t);
				}
			}
		}
	}
};

}

#endif
