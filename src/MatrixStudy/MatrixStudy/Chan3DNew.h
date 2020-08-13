// MatrixStudy.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

#include<eigen/dense>

#define PRINT_DBUEG(X) std::cout << #X":" << std::endl << X << std::endl

// TODO: Reference additional headers your program requires here.
class Chan3DNew {
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

	void Chan3DCalc(
		const Eigen::Index& BSN,
		const Eigen::MatrixXd& BS,
		const Eigen::MatrixXd& R
	)
	{
		//第一次WLS
		//k=X^2+Y^2+Z^2
		Eigen::MatrixXd k;
		k.resize(BSN, 1);
		//BSN为基站个数, BS为基站坐标
		for (Eigen::Index i = 0; i < BSN; i++)
		{
			k(i, 0) = std::pow(BS(0, i), 2) + std::pow(BS(1, i), 2) + std::pow(BS(2, i), 2);
		}
		PRINT_DBUEG(k);

		//h = 1/2(Ri^2-ki+k1)
		Eigen::MatrixXd h;
		h.resize(BSN - 1, 1);
		for (Eigen::Index i = 0; i < BSN - 1; i++)
		{
			h(i, 0) = 0.5 * (std::pow(R(i, 0), 2) - k(i + 1, 0) + k(0, 0));
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
			Ga(i, 3) = -R(i, 0);
		}
		PRINT_DBUEG(Ga);

		//Q为TDOA系统的噪声协方差矩阵
		Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(BSN - 1, BSN - 1);
		PRINT_DBUEG(Q);

		Eigen::MatrixXd Ga_t = Ga.transpose();
		PRINT_DBUEG(Ga_t);
		//Eigen::MatrixXd h_t = h.transpose();
		//PRINT_DBUEG(h_t);
		Eigen::MatrixXd Q_i = Q.inverse();//pinv(Q_pinv, Q);
		PRINT_DBUEG(Q_i);
		Eigen::MatrixXd GatQi = Ga_t * Q_i;
		PRINT_DBUEG(GatQi);
		Eigen::MatrixXd GatQiGa = GatQi * Ga;
		PRINT_DBUEG(GatQiGa);
		Eigen::MatrixXd GatQiGa_i;
		pinv(GatQiGa_i, GatQiGa);
		PRINT_DBUEG(GatQiGa_i);

		//MS与BS距离较近时
		//za = pinv(Ga' * pinv(Q) * Ga) * Ga' * pinv(Q) * h
		Eigen::MatrixXd za0;
		za0 = GatQiGa_i * GatQi * h;
		PRINT_DBUEG(za0);

		//利用这个估算值计算B
		Eigen::MatrixXd B = Eigen::MatrixXd::Identity(BSN - 1, BSN - 1);
		for (Eigen::Index i = 0; i < BSN - 1; i++)
		{
			B(i, i) = std::sqrt(std::pow(BS(0, i+1) - za0(0,0), 2) + std::pow(BS(1, i + 1) - za0(1, 0), 2) + std::pow(BS(2, i + 1) - za0(2, 0), 2));
		}
		PRINT_DBUEG(B);
		
		Eigen::MatrixXd FI = B * Q * B;
		PRINT_DBUEG(FI);

		Eigen::MatrixXd FI_i = FI.inverse();
		PRINT_DBUEG(FI);
		Eigen::MatrixXd GatFIi = Ga_t * FI_i;
		PRINT_DBUEG(GatFIi);
		Eigen::MatrixXd GatFIiGa = GatFIi * Ga;
		PRINT_DBUEG(GatFIiGa);
		Eigen::MatrixXd GatFIiGa_i;
		pinv(GatFIiGa_i, GatFIiGa);
		PRINT_DBUEG(GatFIiGa_i);

		Eigen::MatrixXd za1 = GatFIiGa_i * GatFIi * h;
		PRINT_DBUEG(za1);

		//第二次WLS
		// 第一次LS结果的协方差
		Eigen::MatrixXd CovZa = GatFIiGa_i;
		PRINT_DBUEG(CovZa);
		// 第二次LS
		Eigen::MatrixXd sB = Eigen::MatrixXd::Identity(4, 4);
		for (Eigen::Index i = 0; i < sB.cols(); i++)
		{
			sB(i, i) = za1(i,0);
		}
		PRINT_DBUEG(sB);
		//sFI
		Eigen::MatrixXd sFI = 4 * sB * CovZa * sB;
		PRINT_DBUEG(sFI);

		//sGa
		Eigen::MatrixXd sGa;
		sGa.resize(4, 3);
		sGa << 
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			1, 1, 1;
		PRINT_DBUEG(sGa);
		// sh
		Eigen::MatrixXd sh;
		sh.resize(4, 1);
		sh <<
			std::pow(za1(0, 0), 2),
			std::pow(za1(1, 0), 2),
			std::pow(za1(2, 0), 2),
			std::pow(za1(3, 0), 2);
		PRINT_DBUEG(sh);

		Eigen::MatrixXd sGa_t = sGa.transpose();
		PRINT_DBUEG(sGa_t);
		Eigen::MatrixXd sFI_i = sFI.inverse();
		PRINT_DBUEG(sFI_i);
		Eigen::MatrixXd sGatsFIi = sGa_t * sFI_i;
		PRINT_DBUEG(sGatsFIi);
		Eigen::MatrixXd sGatsFIisGa = sGatsFIi * sGa;
		PRINT_DBUEG(sGatsFIisGa);
		Eigen::MatrixXd sGatsFIisGa_i;// = sGatsFIisGa.inverse();
		pinv(sGatsFIisGa_i, sGatsFIisGa);
		PRINT_DBUEG(sGatsFIisGa_i);

		// 第二次LS结果
		Eigen::MatrixXd za2 = sGatsFIisGa_i * sGatsFIi * sh;
		PRINT_DBUEG(za2);

		Eigen::MatrixXd sZ = za2.array().abs().sqrt();
		PRINT_DBUEG(sZ);

		Eigen::MatrixXd zp = sZ;
		PRINT_DBUEG(zp);
	}
public:
	void Point2D()
	{
		//求二维下一点到原点的距离, Pt(x,y)
		Eigen::Vector2d v1(10.0, 0.0);
		double  res1 = v1.norm();//   等于 sqrt(x^2+y^2) , 即距离
		double  res2 = v1.squaredNorm();//    (x^2+y^2)
		std::cout << "res1=" << res1 << ",res2=" << res2 << std::endl;
	}
	void PointToPoint2D()
	{
		//求二维下一点到另一点的距离, (x1,y1), (x2,y2)
		Eigen::Vector2d v1(10.0, 10.0);
		Eigen::Vector2d v2(5.0, 10.0);
		double  res1 = (v1 - v2).norm();//   等于 sqrt(x^2+y^2) , 即距离
		std::cout << "res1=" << res1 << std::endl;
	}
	void PointToPoint3D()
	{
		//求三维下一点到另一点的距离, (x1,y1,z1), (x2,y2,z2)
		Eigen::Vector3d v1(10.0, 10.0, 10.0);
		Eigen::Vector3d v2(10.0, 10.0, 10.0);
		double  res1 = (v1 - v2).norm();//   等于 sqrt(x^2+y^2) , 即距离
		std::cout << "res1=" << res1 << std::endl;
	}
	void Run()
	{
		system("cls");
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
		Chan3DCalc(BSN, BS, R);
	}

public:
	static Chan3DNew* Inst()
	{
		static Chan3DNew chan3dNewInstance;
		return &chan3dNewInstance;
	}
};