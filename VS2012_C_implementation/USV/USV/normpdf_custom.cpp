#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <mexopencv.hpp>

enum class BsxFunOp { DIVIDE, TIMES };

/*Represents a function equivalent to Matlab's bsxfun*/
cv::Mat Bsxfun(cv::Mat inputMat, cv::Mat bsxParam, BsxFunOp op=BsxFunOp::DIVIDE){
	int rows = inputMat.rows;		
	int cols = inputMat.cols;
	int channels = inputMat.channels();
	
	bsxParam = bsxParam.reshape(1, bsxParam.rows*bsxParam.cols);
	cv::Mat result = inputMat.clone().reshape(1,rows*cols);

	switch (op)
	{
		case BsxFunOp::DIVIDE:
			for (int i = 0; i < result.cols; i++)
			{
				//result.col(i) = result.col(i).mul(bsxParam);
				cv::divide(result.col(i), bsxParam, result.col(i));
			}
			break;

		case BsxFunOp::TIMES:
			for (int i = 0; i < result.cols; i++)
			{			
				cv::multiply(result.col(i), bsxParam, result.col(i));
			}
			break;

		default:
			break;
	}	

	result = result.reshape(channels, rows);
	return result;
}

void mexFunction(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{
	//// Check arguments
	//if (nlhs!=1 || nrhs!=1)		
	//mexErrMsgIdAndTxt("myfunc:invalidArgs", "Wrong number of arguments");
	
	////Example on how to get struct fields
	//tmp = mxGetField(prhs[0], 0, "Mu");		

	try{
		cv::Mat A = MxArray(prhs[0]).toMat();		
		cv::Mat bsxParam = MxArray(prhs[1]).toMat();						
		A = Bsxfun(A,bsxParam);
		plhs[0] = MxArray(A);		
		
	}catch(cv::Exception& ex){
		const char* err_msg = ex.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
}