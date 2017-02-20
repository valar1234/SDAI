/*
 * @author: jjf, Fudan University
 * @date: 2016/10/9
 */
#ifndef __UTILS_H__
#define __UTILS_H__
#include "configure.h"
#include <assert.h>

namespace SDAI
{
	/*
	 * @note: find the catagory, that is the index of the maximum element
	 */
template<int INPUT_DIM>
TYPE_PINT utils_find_category(TYPE_T data[INPUT_DIM])
{
	assert(INPUT_DIM > 0);
	/* set the initial as the first element */
	TYPE_T	max = data[0];
	TYPE_PINT index = 0;

	/* find the maximum element */
	for( int i = 1; i < INPUT_DIM; i++)
	{
#pragma HLS pipeline
		TYPE_T v = data[i];
		if( v > max)
		{
			max = v;
			index = i;
		}
	}
	return index;
}

}



#endif
