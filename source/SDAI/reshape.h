/*
 * @author: jjf, Fudan University
 * @date: 2016/10/13
 */
#ifndef __RESHAPE_H__
#define __RESHAPE_H__
#include "configure.h"
#include "assert.h"

#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

typedef enum{ORDER_X}RESHAPE_MODE;

/*
 * @note: convert 2D array to 1D array
 */
template<int DIM1, int DIM2, RESHAPE_MODE MODE = ORDER_X, int OUTPUT_DIM = DIM1 * DIM2>
class Reshape2D_1D
{
public:
	Reshape2D_1D()
	{
#if DEBUG
		cout <<"Reshape2D_1D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
		cout <<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
	TYPE_T res[OUTPUT_DIM];

public:
	void feedforward(TYPE_T data[DIM1][DIM2])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN || RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
				switch(MODE)
				{
				case ORDER_X: res[i * DIM2 + j] = data[i][j]; break;
				default: assert(0); break;
				}
			}
		}
	}
};

/*
 * @note: convert 3D array to 1D array
 */
template<int DIM1, int DIM2, int DIM3, RESHAPE_MODE MODE = ORDER_X, int OUTPUT_DIM = DIM1 * DIM2 * DIM3>
class Reshape3D_1D
{
public:
	Reshape3D_1D()
	{
#if DEBUG
		cout <<"Reshape3D_1D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
		cout <<"\tDIM3 = " << DIM3 << endl;
		cout <<"\tOUTPUT_DIM = " << OUTPUT_DIM << endl;
#endif
	};

public:
	TYPE_T res[OUTPUT_DIM];

public:
	void feedforward(TYPE_T data[DIM1][DIM2][DIM3])
	{
		for( int i = 0; i < DIM1; i++)
		{
#if RESHAPE_PERF_MODE == PERF_HIGH
#pragma HLS pipeline
#endif
			for( int j = 0; j < DIM2; j++)
			{
#if RESHAPE_PERF_MODE == PERF_MEDIAN
#pragma HLS pipeline
#endif
				for(int k = 0; k < DIM3; k++)
				{
#if RESHAPE_PERF_MODE == PERF_LOW
#pragma HLS pipeline
#endif
					switch(MODE)
					{
					case ORDER_X: res[i * DIM2 * DIM3 + j * DIM3 + k] = data[i][j][k]; break;
					default: assert(0); break;
					}
				}
			}
		}
	}
};

/*
 * @note: convert stream to 1D array
 */
template<int DIM1>
class Reshape_Stream_1D
{
public:
	Reshape_Stream_1D()
	{
#if DEBUG
		cout <<"Reshape_Stream_1D Reshape_Stream_1D......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
#endif
	};

public:
	TYPE_T res[DIM1];

public:
	void feedforward(volatile TYPE_T *data)
	{
		for( int i = 0; i < DIM1; i++)
		{
#pragma HLS pipeline
			res[i] = data[i];
		}

	}
};


/*
 * @note: convert stream to 2D array
 */
template<int DIM1, int DIM2, RESHAPE_MODE MODE = ORDER_X>
class Reshape_Stream_2D
{
public:
	Reshape_Stream_2D()
	{
#if DEBUG
		cout <<"Reshape_Stream_2D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
#endif
	};

public:
	TYPE_T res[DIM1][DIM2];

public:
	void feedforward(volatile TYPE_T *data)
	{
		for( int i = 0; i < DIM1; i++)
		{
#pragma HLS pipeline
			for( int j = 0; j < DIM2; j++)
			{
					switch(MODE)
					{
					case ORDER_X: res[i][j] = data[i * DIM2  + j]; break;
					default: assert(0); break;
					}
			}
		}
	}
};

/*
 * @note: convert stream to 3D array
 */
template<int DIM1, int DIM2, int DIM3, RESHAPE_MODE MODE = ORDER_X>
class Reshape_Stream_3D
{
public:
	Reshape_Stream_3D()
	{
#if DEBUG
		cout <<"Stream_3D Layer......"<<endl;
		cout <<"\tDIM1 = " << DIM1 << endl;
		cout <<"\tDIM2 = " << DIM2 << endl;
		cout <<"\tDIM3 = " << DIM3 << endl;
#endif
	};

public:
	TYPE_T res[DIM1][DIM2][DIM3];

public:
	void feedforward(volatile TYPE_T *data)
	{
		for( int i = 0; i < DIM1; i++)
		{
			for( int j = 0; j < DIM2; j++)
			{
				for(int k = 0; k < DIM3; k++)
				{
#pragma HLS pipeline
					switch(MODE)
					{
					case ORDER_X: res[i][j][k] = data[i * DIM2 * DIM3 + j * DIM3 + k]; break;
					default: assert(0); break;
					}
				}
			}
		}
	}
};

}

#endif
