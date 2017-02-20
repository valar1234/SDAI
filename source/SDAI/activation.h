/*
 * @author: jjf, Fudan University
 * @date: 2016/10/9
 */
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
#include <math.h>
#include <assert.h>
#include "configure.h"

namespace SDAI
{

typedef enum{LINEAR, SIGMOID, HARDSIGMOID, TANH, RELU, LEAKYRELU, THRESHOLDEDRELU, SOFTSIGN, SOFTPLUS, SOFTMAX}ACTIVATION;

#define ABS(x)	((x) > 0 ? (x) : -(x))


/*
 * @note: the activation function of LINEAR
 */
inline TYPE_T activation_linear(TYPE_T x)
{
#pragma HLS INLINE
	return x;
}


/*
 * @note: the activation function of SIGMOID
 */
inline TYPE_T activation_sigmoid(TYPE_T x)
{
#pragma HLS INLINE
	const TYPE_T c = -1.0;
	return 1.0/(1.0 + expf(c * x));
}

/*
 * @note: the activation function of HARD SIGMOID
 */
inline TYPE_T activation_hardsigmoid(TYPE_T x)
{
#pragma HLS INLINE
	const TYPE_T m = 0.2;
	const TYPE_T n = 0.5;
	TYPE_T	v = x * m + n;

	if( v >= 1.0)
		return 1.0;
	else if( v <= 0.0)
		return 0.0;
	else
		return v;
}

/*
 * @note: the activation function of RELU
 */
inline TYPE_T activation_relu(TYPE_T x)
{
#pragma HLS INLINE
	const TYPE_T t = 0;
	return x >= t ? x : t;
}


/*
 * @note: the activation function of TANH
 */
inline TYPE_T activation_tanh(TYPE_T x)
{
#pragma HLS INLINE
	const TYPE_T c = 2.0;
	return 1.0 - 2.0/(expf(c * x) + 1.0);
}

/*
 * @note: the activation function of softsign
 */
inline TYPE_T activation_softsign(TYPE_T x)
{
#pragma HLS INLINE
	if( x > 0)
		return x/(1 + x);
	else
		return x/(1 - x);
}

/*
 * @note: the activation function of softplus
 */
inline TYPE_T activation_softplus(TYPE_T x)
{
#pragma HLS INLINE
	return logf( 1 + expf(x));
}

/*
 * @note: the activation function of softmax
 */
template<int OUTPUT_DIM>
void activation_softmax(TYPE_T in[OUTPUT_DIM])
{
	/* find the maximum */
	TYPE_T max = in[0];
	for(int i = 1; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		if( in[i] > max)
			max = in[i];
	}

	/* calculate the new value */
	TYPE_T val[OUTPUT_DIM];
	TYPE_T sum = 0;
	for( int i = 0; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		val[i] = expf(in[i] - max);
		sum += val[i];
	}

	/* calculate the result */
	for( int i = 0; i < OUTPUT_DIM; i++)
	{
#pragma HLS pipeline
		in[i] = val[i]/sum;
	}

}


/*
 * @note: some Advanced Activation Functions
 */
/*
 * @note: the LeakyRelU
 */
inline TYPE_T activation_leakyrelu(TYPE_T x)
{
	const TYPE_T alpha = 0.3;
	if( x < 0.0)
		return alpha * x;
	else
		return x;
}

/*
 * @note: the Thresholded ReLU
 */
inline TYPE_T activation_thresholdedrelu(TYPE_T x)
{
#pragma HLS INLINE
	const TYPE_T theta = 1.0;
	const TYPE_T v = 0.0;
	return x > theta ? x : v;
}


/*
 * @note: the  activation function
 */
template<ACTIVATION AC_FN>
inline TYPE_T activation_fn(TYPE_T x)
{
#pragma HLS INLINE
	TYPE_T res;
	/* calculate the activation function */
	switch( AC_FN )
	{
	case LINEAR: 		res = activation_linear( x ); 		break;
	case SIGMOID: 		res = activation_sigmoid( x ); 		break;
	case HARDSIGMOID: 	res = activation_hardsigmoid( x ); 	break;
	case TANH: 			res = activation_tanh( x ); 		break;
	case RELU: 			res = activation_relu( x ); 		break;
	case SOFTSIGN: 		res = activation_softsign( x ); 	break;
	case SOFTPLUS: 		res = activation_softplus( x ); 	break;
	case SOFTMAX: 		res = activation_linear( x ); 		break;
	case LEAKYRELU:		res = activation_leakyrelu( x ); 	break;
	case THRESHOLDEDRELU:	res = activation_thresholdedrelu( x ); 	break;
	default: assert( 0 ); break;
	}
	return res;
}

}

#endif
