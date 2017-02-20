/*
 * @author: jjf, Fudan University
 * @date: 2016/11/13
 */
#ifndef __MEM_H__
#define __MEM_H__
#include "activation.h"
#include "configure.h"
#include <assert.h>
#include "reshape.h"
#if DEBUG
#include <iostream>
using namespace std;
#endif

namespace SDAI
{

template< int DIM1>
class LineBuffer1D;

template<int DIM1, int DIM2, int SHIFT_ROW>
class LineBuffer2D;

template <int DIM1, int DIM2, int DIM3, int LineDIM2, int SHIFT_ROW, int SHIFT_COL>
class WindowBuffer3D;

template<int DIM1, int DIM2, int DIM3, int SHIFT_ROW>
class LineBuffer3D;


/*
 * @note: 1D line buffer
 */
template< int DIM1>
class LineBuffer1D
{
public:
	LineBuffer1D()
	{
	}
public:
	TYPE_T	val[DIM1];
public:
	/*
	 * @note: fill the buffer
	 */
	void fill(volatile TYPE_T *data)
	{
#pragma HLS inline
		for( int i = 0; i < DIM1; i++)
		{
#pragma HLS pipeline
			val[i] = data[i];
		}
	}
	/*
	 * @note: get the value
	 */
	TYPE_T& getval(int dim1)
	{
#pragma HLS inline
		return val[dim1];
	}
};


/*
 * @note: define the 3D window buffer
 */
template <int DIM1, int DIM2, int DIM3, int LineDIM2, int SHIFT_ROW, int SHIFT_COL>
class WindowBuffer3D
{
public:
	WindowBuffer3D()
	{
		assert(DIM2 >= SHIFT_COL);
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
#pragma HLS ARRAY_PARTITION variable=val dim=2 complete
#pragma HLS ARRAY_PARTITION variable=val dim=3 complete
	}

public:
	TYPE_T val[DIM1][DIM2][DIM3];

	/*
	 * @note: fill the window buffer
	 */
	void fill(LineBuffer3D<DIM1, LineDIM2, DIM3, SHIFT_ROW> &l_buffer, int dim2)
	{
		for( int i = 0; i < DIM1; i++)
		{
			for( int j = 0; j < DIM2; j++)
			{
				for( int k = 0; k < DIM3; k++)
				{
					val[i][j][k]= l_buffer.getval(i, j + dim2, k);
				}
			}
		}
	}

	/*
	 * @note: get value function
	 */
	TYPE_T& getval(int dim1, int dim2, int dim3)
	{
#pragma HLS inline
		return val[dim1][dim2][dim3];
	}

	/*
	 * @note: shit left the element
	 */
	void shift_left()
	{
#pragma HLS inline

		for(int i = 0; i < DIM1; i++)
		{
			for(int j = 0; j < DIM2 - SHIFT_COL; j++)
			{
				for( int k = 0; k < DIM3; k++)
				{
					val[i][j][k] = val[i][j + SHIFT_COL][k];
				}
			}
		}
	}

	/*
	 * @note: insert data from the right side
	 */
	void insert_right(LineBuffer3D<DIM1, LineDIM2, DIM3, SHIFT_ROW> &l_buffer, int dim2)
	{
#pragma HLS inline

		for( int i = 0; i < DIM1; i++)
		{
			for( int j = SHIFT_COL - 1; j >= 0; j--)
			{
				for( int k = 0; k < DIM3; k++)
				{
					val[i][DIM2 - 1 - j][k] = l_buffer.getval(i, dim2 + j, k);
				}
			}
		}
	}
};

/*
 * @note: define the 2D stream line buffer
 */
template<int DIM1, int DIM2, int SHIFT_ROW>
class LineBuffer2D
{
public:
	LineBuffer2D()
	{
		assert(DIM1 >= SHIFT_ROW);
#pragma HLS array_reshape variable=val dim=1
#pragma HLS dependence variable=val inter false
#pragma HLS dependence variable=val intra false
	}
public:
	TYPE_T val[DIM1][DIM2];

public:

	/*
	 * @note: fill the arrays
	 */
	void fill(volatile TYPE_T *data)
	{
		for(int i = 0; i < DIM1; i++)
		{
#pragma HLS pipeline
			for(int j = 0; j < DIM2; j++)
			{
				val[i][j] = data[i * DIM2  + j];
			}
		}
	}

	/*
	 * @note: fill the new line
	 */
	void fill_line(volatile TYPE_T *data)
	{
		for( int i = SHIFT_ROW - 1; i >= 0; i--)
		{
#pragma HLS pipeline
			for( int j = 0; j < DIM2; j++)
			{
				val[DIM1 - 1 - i][j] = data[(SHIFT_ROW - 1 - i) * DIM2  + j];
			}
		}
	}


	/*
	 * @note: shift down function
	 */
	void shift_up()
	{
		for(int i = 0; i < DIM1 - SHIFT_ROW; i++)
		{
#pragma HLS unroll
			for(int dim2 = 0; dim2 < DIM2; dim2++)
			{
#pragma HLS unroll
				val[i][dim2] = val[i + SHIFT_ROW][dim2];
			}
		}
	}


	/*
	 * @note: get the value
	 */
	TYPE_T& getval(int dim1, int dim2)
	{
#pragma HLS inline
		return val[dim1][dim2];
	}

};


/*
 * @note: define the 3D stream line buffer
 */
template<int DIM1, int DIM2, int DIM3, int SHIFT_ROW>
class LineBuffer3D
{
public:
	LineBuffer3D()
	{
		assert(DIM1 >= SHIFT_ROW);
#pragma HLS array_reshape variable=val dim=1
#pragma HLS dependence variable=val inter false
#pragma HLS dependence variable=val intra false
	}
public:
	TYPE_T val[DIM1][DIM2][DIM3];

public:

	/*
	 * @note: fill the arrays
	 */
	void fill(volatile TYPE_T *data)
	{
		for(int i = 0; i < DIM1; i++)
		{
			for(int j = 0; j < DIM2; j++)
			{
				for( int k = 0; k < DIM3; k++)
				{
#pragma HLS pipeline
					val[i][j][k] = data[i * DIM2 * DIM3 + j * DIM3 + k];
				}
			}
		}
	}

	/*
	 * @note: fill the new line
	 */
	void fill_line(volatile TYPE_T *data)
	{
#pragma HLS inline

		for( int i = SHIFT_ROW - 1; i >= 0; i--)
		{
			for( int j = 0; j < DIM2; j++)
			{
				for( int k = 0; k < DIM3; k++)
				{
#pragma HLS pipeline
					val[DIM1 - 1 - i][j][k] = data[(SHIFT_ROW - 1 - i) * DIM2 * DIM3 + j * DIM3 + k];
				}
			}
		}
	}


	/*
	 * @note: shift down function
	 */
	void shift_up()
	{
#pragma HLS inline

		for(int i = 0; i < DIM1 - SHIFT_ROW; i++)
		{
			for(int dim2 = 0; dim2 < DIM2; dim2++)
			{
				for(int dim3 = 0; dim3 < DIM3; dim3++)
				{
					val[i][dim2][dim3] = val[i + SHIFT_ROW][dim2][dim3];
				}
			}
		}
	}


	/*
	 * @note: get the value
	 */
	TYPE_T& getval(int dim1, int dim2, int dim3)
	{
#pragma HLS inline
		return val[dim1][dim2][dim3];
	}

};

}

#endif
