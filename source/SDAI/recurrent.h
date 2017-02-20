/*
 * @author: jjf, Fudan University
 * @date: 2016/11/2
 */
#ifndef __RECURRENT_H__
#define __RECURRENT_H__
#include "activation.h"
#include "configure.h"
#include <assert.h>


#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{
/*
 * @note: define the Simple RNN based on Elman Neural Network, http://outlace.com/Simple-Recurrent-Neural-Network/
 */
template<int INPUT_LENGTH, int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN >
class SimpleRNN
{
public:
	SimpleRNN(const TYPE_T *WEIGHT)
	{
#if DEBUG
		cout<<"SimpleRNN Layer......"<<endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
		cout<<"\tINPUT_LENGTH = " << INPUT_LENGTH << endl;
#endif

		/* initialize the weight */
		for( int i = 0; i < OUTPUT_DIM + INPUT_DIM + 1; i++)
		{
			for(int j = 0; j < OUTPUT_DIM; j++)
			{
				weight[i][j] = WEIGHT[i * OUTPUT_DIM + j];
			}
		}
	}

public:
	TYPE_T	weight[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	res[OUTPUT_DIM];

public:
	/*
	 *@note: the feedforward function
	 */
	void feedforward(TYPE_T data[INPUT_LENGTH][INPUT_DIM])
	{
		/* initialize the context */
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#pragma HLS pipeline
			res[i] = 0;
		}

		for(int i = 0; i < INPUT_LENGTH; i++)
		{
#if RECURRENT_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j <OUTPUT_DIM; j++)
			{
#if RECURRENT_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* add the bias */
				TYPE_T tmp = weight[OUTPUT_DIM + INPUT_DIM][j];

				/* add the input weight */
				for( int k = 0; k < INPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					tmp += data[i][k] * weight[k][j];
				}

				/* add the memory cell weight */
				for( int k = 0; k < OUTPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					tmp += res[k] * weight[k + INPUT_DIM][j];
				}

				/* calculate the activation function */
				res[j] = activation_fn<AC_FN>(tmp);
			}

			/* for the activation of softmax */
			if( AC_FN == SOFTMAX )
			{
				activation_softmax<OUTPUT_DIM>(res);
			}
		}
	}
};



/*
 * @note: define the Gated Recurrent Neural Network, https://arxiv.org/pdf/1412.3555v1.pdf
 */
template<int INPUT_LENGTH, int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN = TANH, ACTIVATION INNER_AC_FN = SIGMOID>
class GRU
{
public:
	GRU(const TYPE_T *WEIGHT_Z, const TYPE_T *WEIGHT_R, const TYPE_T *WEIGHT_H)
	{
#if DEBUG
		cout<<"GRU Layer......"<<endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
		cout<<"\tINPUT_LENGTH = " << INPUT_LENGTH << endl;
#endif

		/* initialize the weights */
		for( int i = 0; i < OUTPUT_DIM + INPUT_DIM + 1; i++)
		{
			for( int j = 0; j < OUTPUT_DIM; j++)
			{
				weight_z[i][j] = WEIGHT_Z[i*OUTPUT_DIM + j];
				weight_r[i][j] = WEIGHT_R[i*OUTPUT_DIM + j];
				weight_h[i][j] = WEIGHT_H[i*OUTPUT_DIM + j];
			}
		}
	}
public:
	TYPE_T	weight_z[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	weight_r[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	weight_h[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	res[OUTPUT_DIM];
	TYPE_T	rr[OUTPUT_DIM];
	TYPE_T	zz[OUTPUT_DIM];
	TYPE_T	rh[OUTPUT_DIM];

public:
	/*
	 * @note: the feedforward function
	 */
	void feedforward(TYPE_T data[INPUT_LENGTH][INPUT_DIM])
	{
		/* initialize the context */
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
#pragma HLS pipeline
			res[i] = 0;
			rr[i] = 0;
			zz[i] = 0;
			rh[i] = 0;
		}

		for(int i = 0; i < INPUT_LENGTH; i++)
		{
#if RECURRENT_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j <OUTPUT_DIM; j++)
			{
#if RECURRENT_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* add the bias */
				TYPE_T r = weight_r[OUTPUT_DIM + INPUT_DIM][j];
				TYPE_T z = weight_z[OUTPUT_DIM + INPUT_DIM][j];

				/* add the input data weight */
				for( int k = 0; k < INPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					TYPE_T xk = data[i][k];
					r += weight_r[k][j] * xk;
					z += weight_z[k][j] * xk;
				}

				for( int k = 0; k < OUTPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					TYPE_T prev_h = res[k];
					r += prev_h * weight_r[k + INPUT_DIM][j];
					z += prev_h * weight_z[k + INPUT_DIM][j];
				}

				/* calculate the inner activation function */
				rr[j] = activation_fn<INNER_AC_FN>(r);
				zz[j] = activation_fn<INNER_AC_FN>(z);

				/* calculate the r * h */
				rh[j] = rr[j] * res[j];
			}

			/* calculate the new h*/
			for( int j = 0; j < OUTPUT_DIM; j++)
			{
#if RECURRENT_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* add the W and bias*/
				TYPE_T h_new = weight_h[OUTPUT_DIM + INPUT_DIM][j];
				for(int k = 0; k < INPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					h_new += weight_h[k][j] * data[i][k];
				}

				for( int k = 0; k < OUTPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					h_new += rh[j] * weight_h[k + INPUT_DIM][j];
				}


				TYPE_T hh_new;
				hh_new = activation_fn<AC_FN>(h_new);

				/* calculate and update the final result */
				res[j] = (1 - zz[j]) * res[j] + zz[j] * hh_new;
			}

			/* for the activation of softmax */
			if( AC_FN == SOFTMAX )
			{
				activation_softmax<OUTPUT_DIM>(res);
			}
		}
	}

};

/*
 * @note: define the Long Short Term Memory(LSTM) Neural Network, http://deeplearning.net/tutorial/lstm.html
 */
template<int INPUT_LENGTH, int INPUT_DIM, int OUTPUT_DIM, ACTIVATION AC_FN = TANH, ACTIVATION INNER_AC_FN = SIGMOID>
class LSTM
{
public:
	LSTM(const TYPE_T *WEIGHT_I, const TYPE_T *WEIGHT_C, const TYPE_T *WEIGHT_F, const TYPE_T *WEIGHT_O)
	{
#pragma HLS ARRAY_PARTITION variable=weight_o block factor=4 dim=1


#if DEBUG
		cout<<"LSTM Layer......"<<endl;
		cout<<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
		cout<<"\tINPUT_LENGTH = " << INPUT_LENGTH << endl;
#endif

		/* initialize the weights */
		for( int i = 0; i < OUTPUT_DIM + INPUT_DIM + 1; i++)
		{
			for(int j = 0; j < OUTPUT_DIM; j++)
			{
				weight_i[i][j] = WEIGHT_I[i*OUTPUT_DIM + j];
				weight_c[i][j] = WEIGHT_C[i*OUTPUT_DIM + j];
				weight_f[i][j] = WEIGHT_F[i*OUTPUT_DIM + j];
				weight_o[i][j] = WEIGHT_O[i*OUTPUT_DIM + j];
			}
		}
	}

public:
	TYPE_T weight_i[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T weight_c[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T weight_f[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T weight_o[OUTPUT_DIM + INPUT_DIM + 1][OUTPUT_DIM];
	TYPE_T	res[OUTPUT_DIM];
	TYPE_T	ct[OUTPUT_DIM];

public:
	/*
	 * @note: the feed forward function
	 */
	void feedforward(TYPE_T data[INPUT_LENGTH][INPUT_DIM])
	{
		/* initialize the context */
		for( int i = 0; i < OUTPUT_DIM; i++)
		{
			res[i] = 0;
			ct[i] = 0;
		}

		for( int i = 0; i < INPUT_LENGTH; i++)
		{
#if RECURRENT_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < OUTPUT_DIM; j++)
			{
#if RECURRENT_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				/* calculate the W and b*/
				TYPE_T	it =  weight_i[ OUTPUT_DIM + INPUT_DIM][j];
				TYPE_T	cc =  weight_c[ OUTPUT_DIM + INPUT_DIM][j];
				TYPE_T 	ft =  weight_f[ OUTPUT_DIM + INPUT_DIM][j];
				TYPE_T  ot =  weight_o[ OUTPUT_DIM + INPUT_DIM][j];
				for( int k = 0; k < INPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					TYPE_T xk = data[i][k];
					it += (weight_i[k][j] * xk);
					cc += (weight_c[k][j] * xk);
					ft += (weight_f[k][j] * xk);
					ot += (weight_o[k][j] * xk);
				}


				/* calculate the U */
				for(int k = 0; k < OUTPUT_DIM; k++)
				{
#if RECURRENT_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					TYPE_T	pre_h = res[k];
					it += pre_h * weight_i[k + INPUT_DIM][j];
					cc += pre_h * weight_c[k + INPUT_DIM][j];
					ft += pre_h * weight_f[k + INPUT_DIM][j];
					ot += pre_h * weight_o[k + INPUT_DIM][j];
				}


				/* the inner activation function */
				TYPE_T it_o, ft_o, ot_o;
				it_o = activation_fn<INNER_AC_FN>(it);
				ft_o = activation_fn<INNER_AC_FN>(ft);
				ot_o = activation_fn<INNER_AC_FN>(ot);

				/* the activation function */
				TYPE_T cc_o;
				cc_o = activation_fn<AC_FN>(cc);

				/* calculate the memory cell output */
				TYPE_T	ct_new = it_o * cc_o + ft_o * ct[j];

				/* the activation function */
				TYPE_T ct_o;
				ct_o = activation_fn<AC_FN>(ct_new);

				/* calculate the result */
				res[j] = ot_o * ct_o;
				ct[j] = ct_new;

			}

			/* for the activation of softmax */
			if( AC_FN == SOFTMAX )
			{
				activation_softmax<OUTPUT_DIM>(res);
			}
		}
	}
};



}

#endif
