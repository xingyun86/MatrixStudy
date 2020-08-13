// MatrixStudy.cpp : Defines the entry point for the application.
//

#include "MatrixStudy.h"
#include "Chan3D.h"
#include "Chan3DNew.h"

#include <time.h>

int main(int argc, char ** argv)
{
	std::cout << "Hello CMake." << std::endl;

	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);
	m = (m + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
	size_t s = m.size();
	std::cout << "m =" << std::endl << m << std::endl;
	Eigen::VectorXd v(3);
	v << 1, 2, 3;
	std::cout << "m * v =" << std::endl << m * v << std::endl;

	//定义 3x3 矩阵 m3x3 为单位矩阵
	Eigen::MatrixXd m3x3 = Eigen::MatrixXd::Identity(3, 3);
	PRINT_DBUEG(m3x3);
	//定义 4x4 矩阵 m4x4 为单位矩阵
	Eigen::MatrixXd m4x4 = Eigen::MatrixXd::Identity(4, 4);
	PRINT_DBUEG(m4x4);
	//定义 5x5 矩阵 m5x5 为单位矩阵
	Eigen::MatrixXd m5x5 = Eigen::MatrixXd::Identity(5, 5);
	PRINT_DBUEG(m5x5);
	
	Chan3DNew::Inst()->Run();

	return 0;
}
