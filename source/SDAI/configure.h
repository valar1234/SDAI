/*
 * @author: jjf, Fudan University
 * @date: 2016/10/9
 */
#ifndef __CONFIGURE_H__
#define __CONFIGURE_H__
#include <ap_fixed.h>

namespace SDAI
{
/*
 * @note: configure the performance priority
 */



/*
 * @note: configure the performance for each layer, it can be configured as PERF_HIGH, PERF_MEDIAN and PERF_LOW
 */
#define PERF_HIGH								3
#define PERF_MEDIAN								2
#define PERF_LOW								1

#define DENSE_PERF_MODE							PERF_HIGH
#define CONVOLUTION1D_PERF_MODE					PERF_HIGH
#define CONVOLUTION2D_PERF_MODE					PERF_HIGH
#define EMBEDDING_PERF_MODE						PERF_HIGH
#define POOLING1D_PERF_MODE						PERF_HIGH
#define POOLING2D_PERF_MODE						PERF_HIGH
#define RECURRENT_PERF_MODE						PERF_MEDIAN
#define RESHAPE_PERF_MODE						PERF_HIGH
#define UTILS_PERF_MODE							PERF_HIGH

/*
 * @note: configure the memory optimization method for AXI master interface
 */
#define	OPT_NONE								0
#define	OPT_MEM									1
#define OPT_BUFFER								2

#define CONVOLUTION1D_OPT_MODE					OPT_MEM
#define POOLING1D_OPT_MODE						OPT_NONE
#define CONVOLUTION2D_OPT_MODE					OPT_MEM
#define POOLING2D_OPT_MODE						OPT_MEM

/*
 * @note: the Debug switch
 */
#define	DEBUG			0

/*
 * @note: user define data type
 */
typedef			unsigned int				TYPE_PINT;
//typedef		double						TYPE_T;
typedef			float						TYPE_T;
//typedef		ap_fixed<15, 6, AP_TRN_ZERO>	TYPE_T;
//typedef		ap_fixed<20, 12, AP_TRN_ZERO>	TYPE_T;


}
#endif
