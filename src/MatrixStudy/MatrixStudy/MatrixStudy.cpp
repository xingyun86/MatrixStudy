// MatrixStudy.cpp : Defines the entry point for the application.
//

#include "MatrixStudy.h"

#include<Eigen/Dense>
#include <time.h>

#define PRINT_DBUEG(X) std::cout << #X":" << std::endl << X << std::endl

//求解协方差矩阵
Eigen::MatrixXd& cov(Eigen::MatrixXd& outMatrix, const Eigen::MatrixXd& inMatrix)
{
	Eigen::MatrixXd mean_matrix = inMatrix.colwise().mean();
	Eigen::RowVectorXd mean_row_vector(Eigen::RowVectorXd::Map(mean_matrix.data(), inMatrix.cols()));

	Eigen::MatrixXd zeroMeanMat = inMatrix;
	zeroMeanMat.rowwise() -= mean_row_vector;
	if (inMatrix.rows() == 1)
	{
		outMatrix = (zeroMeanMat.adjoint() * zeroMeanMat) / double(inMatrix.rows());
	}
	else
	{
		outMatrix = (zeroMeanMat.adjoint() * zeroMeanMat) / double(inMatrix.rows() - 1);
	}
	return outMatrix;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//伪逆矩阵(Moore-Penrose pseudoinverse)A定义：
//A+=VD+UT,其中，U，D和V是矩阵A奇异值分解后得到的矩阵。对角矩阵D的伪逆D+是非零元素取倒数之后再转置得到的。
//
Eigen::MatrixXd& pinv(Eigen::MatrixXd& outMatrix, const Eigen::MatrixXd& inMatrix)
{
	const double PINV_TOLERANCE = 1.e-6; // choose your tolerance wisely!
	Eigen::JacobiSVD<Eigen::MatrixXd> jacobi_svd(inMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::VectorXd singular_values = jacobi_svd.singularValues();
	Eigen::VectorXd v_singular_values = jacobi_svd.singularValues();

	for (Eigen::Index i = 0; i < jacobi_svd.cols(); ++i)
	{
		if (v_singular_values(i) > PINV_TOLERANCE)
		{
			singular_values(i) = 1.0 / v_singular_values(i);
		}
		else
		{
			singular_values(i) = 0;
		}
	}
	outMatrix = (jacobi_svd.matrixV() * singular_values.asDiagonal() * jacobi_svd.matrixU().transpose());
	return outMatrix;
}

void Chan3D(
	const Eigen::Index& BSN,
	const Eigen::MatrixXd& BS,
	const Eigen::MatrixXd& R
)
{
	//第一次WLS
	//k=X^2+Y^2+Z^2
	Eigen::MatrixXd k;
	k.resize(BSN,1);
	//BSN为基站个数, BS为基站坐标
	for (Eigen::Index i = 0; i < BSN; i++)
	{
		k(i,0) = std::pow(BS(0, i), 2) + std::pow(BS(1, i), 2) + std::pow(BS(2, i), 2);
	}
	PRINT_DBUEG(k);

	//h = 1/2(Ri^2-ki+k1)
	Eigen::MatrixXd h;
	h.resize(BSN - 1, 1);
	for (Eigen::Index i = 0; i < BSN - 1; i++)
	{
		h(i,0) = 0.5 * (std::pow(R(i,0),2) - k(i + 1,0) + k(0, 0));
	}
	PRINT_DBUEG(h);

	//Ga = [Xi,Yi,Zi,Ri]
	Eigen::MatrixXd Ga;
	Ga.resize(BSN - 1, 4);
	for (Eigen::Index i = 0; i < BSN - 1; i++)
	{
		Ga(i, 0) = -BS(0, i + 1);
		Ga(i, 1) = -BS(1, i + 1);
		Ga(i, 2) = -BS(2, i + 1);
		Ga(i, 3) = -R(i,0);
	}
	PRINT_DBUEG(Ga);

	//Q为TDOA系统的协方差矩阵
	Eigen::MatrixXd Q;
	//Q.resize(1, R.cols());
	cov(Q, R);
	PRINT_DBUEG(Q);

	Eigen::MatrixXd Ga_t = Ga.transpose();
	PRINT_DBUEG(Ga_t);
	//Eigen::MatrixXd h_t = h.transpose();
	//PRINT_DBUEG(h_t);
	Eigen::MatrixXd Q_i = Q.inverse();//pinv(Q_pinv, Q);
	PRINT_DBUEG(Q_i);
	Eigen::MatrixXd GatQi = Ga_t * Q_i(0, 0);
	PRINT_DBUEG(GatQi);
	Eigen::MatrixXd GatQiGa = GatQi * Ga;
	PRINT_DBUEG(GatQiGa);
	Eigen::MatrixXd GatQiGa_i;
	pinv(GatQiGa_i, GatQiGa);
	PRINT_DBUEG(GatQiGa_i);
	
	//MS与BS距离较近时
	//za = pinv(Ga' * pinv(Q) * Ga) * Ga' * pinv(Q) * h
	Eigen::MatrixXd za;
	za = GatQiGa_i * GatQi * h;
	PRINT_DBUEG(za);

	//第二次WLS
	double X1 = BS(0, 0);
	double Y1 = BS(0, 0);
	double Z1 = BS(0, 0);
	Eigen::MatrixXd h2;
	h2.resize(4, 1);
	h2 <<
		std::pow(za(0, 0) - X1, 2),
		std::pow(za(1, 0) - Y1, 2),
		std::pow(za(2, 0) - Z1, 2),
		std::pow(za(3, 0), 2);
	PRINT_DBUEG(h2);

	Eigen::MatrixXd Ga2;
	Ga2.resize(4,3);
	Ga2 <<
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1;
	PRINT_DBUEG(Ga2);

	Eigen::MatrixXd B2;
	B2.resize(4, 4);
	B2 <<
		za(0, 0) - X1, 0, 0, 0,
		0, za(1, 0) - Y1, 0, 0,
		0, 0, za(2, 0) - Z1, 0,
		0, 0, 0, za(3, 0);
	PRINT_DBUEG(B2);

	Eigen::MatrixXd Ga2_t = Ga2.transpose();
	PRINT_DBUEG(Ga2_t);
	Eigen::MatrixXd B2_i = B2.inverse();
	PRINT_DBUEG(B2_i);
	Eigen::MatrixXd Ga2tB2iGatQiGaB2i = Ga2_t * B2_i * Ga_t * Q_i(0, 0) * Ga * B2_i;
	PRINT_DBUEG(Ga2tB2iGatQiGaB2i);
	Eigen::MatrixXd Ga2tB2iGatQiGaB2iGa2 = Ga2tB2iGatQiGaB2i * Ga2;
	PRINT_DBUEG(Ga2tB2iGatQiGaB2iGa2);
	Eigen::MatrixXd Ga2tB2iGatQiGaB2iGa2_i;
	pinv(Ga2tB2iGatQiGaB2iGa2_i, Ga2tB2iGatQiGaB2iGa2);
	PRINT_DBUEG(Ga2tB2iGatQiGaB2iGa2_i);

	//距离较远时
	//za2 = pinv(Ga2' * pinv(B2) * Ga' * pinv(Q) * Ga * pinv(B2) * Ga2) * (Ga2' * pinv(B2) * Ga' * pinv(Q) * Ga * pinv(B2)) * h2;
	Eigen::MatrixXd za2;
	za2 = Ga2tB2iGatQiGaB2iGa2_i * Ga2tB2iGatQiGaB2i * h2;
	PRINT_DBUEG(za2);

	Eigen::MatrixXd zp;
	zp.resize(3, 1);
	zp(0, 0) = std::pow(std::abs(za2(0, 0)), 0.5) + X1;
	zp(1, 0) = std::pow(std::abs(za2(1, 0)), 0.5) + Y1;
	zp(2, 0) = std::pow(std::abs(za2(2, 0)), 0.5) + Z1;
	PRINT_DBUEG(zp);
}

int main(int argc, char ** argv)
{
	{
		Eigen::Index x = 2;
		auto a = std::pow(x, 2);
		//测试区域
		Eigen::MatrixXd R(3, 3);
		R << 1, 122, 3, 4, 55, 6, 7, 23, 9;
		Eigen::MatrixXd Q;
		cov(Q, R);
		std::cout << Q << std::endl;

		Eigen::MatrixXd R1(3,1);
		R1 << 1, 122, 3;
		Eigen::MatrixXd Q1;
		cov(Q1, R1);
		std::cout << Q1 << std::endl;

		Eigen::MatrixXd Rpinv;
		pinv(Rpinv, R);
		std::cout << "Rpinv:" << std::endl << Rpinv << std::endl;

		//system("pause");
	}
	std::cout << "Hello CMake." << std::endl;

	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);
	m = (m + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
	size_t s = m.size();
	std::cout << "m =" << std::endl << m << std::endl;
	Eigen::VectorXd v(3);
	v << 1, 2, 3;
	std::cout << "m * v =" << std::endl << m * v << std::endl;
	{
		//求二维下一点到原点的距离, Pt(x,y)
		Eigen::Vector2d v1(10.0, 0.0);
		double  res1 = v1.norm();//   等于 sqrt(x^2+y^2) , 即距离
		double  res2 = v1.squaredNorm();//    (x^2+y^2)
		std::cout << "res1=" << res1 << ",res2=" << res2 << std::endl;
	}
	{
		//求二维下一点到另一点的距离, (x1,y1), (x2,y2)
		Eigen::Vector2d v1(10.0, 10.0);
		Eigen::Vector2d v2(5.0, 10.0);
		double  res1 = (v1 - v2).norm();//   等于 sqrt(x^2+y^2) , 即距离
		std::cout << "res1=" << res1 << std::endl;
	}
	{
		//求三维下一点到另一点的距离, (x1,y1,z1), (x2,y2,z2)
		Eigen::Vector3d v1(10.0, 10.0, 10.0);
		Eigen::Vector3d v2(10.0, 10.0, 10.0);
		double  res1 = (v1 - v2).norm();//   等于 sqrt(x^2+y^2) , 即距离
		std::cout << "res1=" << res1 << std::endl;
	}
	{
		system("cls");
		srand(time(nullptr));
		//定义光速
		const double C0 = 3 * 10e9;//(km/s)
		//基站数量
		Eigen::Index BSN = 6;
		//基站坐标
		Eigen::MatrixXd BS;
		BS.resize(3, BSN);
		BS <<
			0, 86.602540378443860, 43.301270189221930, -43.301270189221930, -86.602540378443860, -43.301270189221930,
			0, 0, 75, 75, 0, -75,
			0, 0, 75, 75, 0, -75;
		PRINT_DBUEG(BS);

		//基站时钟数据
		Eigen::MatrixXd R0;
		R0.resize(BSN - 1, 1);
		R0 <<
			18.484491763848773, 13.355091018230437, 45.510937389367430, 71.189747374800870, 1.134607886550090e+02;
		PRINT_DBUEG(R0);

		Eigen::MatrixXd R;
		R.resize(BSN - 1, 1);
		/*for (Eigen::Index i = 0; i < R.size() - 1; i++)
		{
			R(i,0) = R0(Eigen::Index(i) + 1, 0) - R0(0, 0);
		}*/
		R << 17.3477082749703, 13.0324148452678, 44.5503414498834, 74.2974876429252, 114.776084308268;

		PRINT_DBUEG(R);
		Chan3D(BSN, BS, R);
	}
	return 0;
}
