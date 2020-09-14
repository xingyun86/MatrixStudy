// KalmanStudy.cpp : Defines the entry point for the application.
//

#include "KalmanStudy.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>//包含Eigen矩阵运算库，用于矩阵计算
#include <cmath>
#include <limits>//用于生成随机分布数列

using namespace std;
using Eigen::MatrixXd;


class KalmanFilterHelper {
#if defined(DEBUG) || defined(_DEBUG)
#define KFH_PRINT_BLOCK(X) std::cout << #X "[" << X.rows() << "," << X.cols() << "]" << std::endl
#endif // DEBUG || _DEBUG

private:
	MatrixXd X;//最优值(N*1阶矩阵)
	MatrixXd F;//状态转移矩阵(N*N阶方阵)
	MatrixXd B;//控制增益矩阵
	MatrixXd H;//测量矩阵(N/2*N阶矩阵)
	/// <summary>
	/// 矩阵是卡尔曼滤波中比较难确定的一个量，
	/// 一般有两种思路：
	/// 一是在某些稳定的过程可以假定它是固定的矩阵，
	///     通过寻找最优的Q值使滤波器获得更好的性能，
	///     这是调整滤波器参数的主要手段，Q一般是对角阵，
	///     且对角线上的值很小，便于快速收敛；
	/// 二是在自适应卡尔曼滤波（AKF）中Q矩阵是随时间变化的
	/// </summary>
	MatrixXd Q;//过程激励噪声的协方差(即状态转移矩阵与实际过程之间的误差)
	MatrixXd R;//测量噪声协方差(取值过大或过小都会使滤波效果变差，且R取值越小收敛越快)
	MatrixXd P;//某个时刻的先验估计协方差和后验估计协方差
	MatrixXd K;//卡尔曼增益(滤波的中间结果)
	MatrixXd I;//N*N阶单位矩阵
	double T = 0.0;//控制周期
public:
	MatrixXd _X;//预测值(N*1阶矩阵)
	MatrixXd _Z;//测量值(N/2*1阶矩阵)
	double _U = 0.0; //某个时刻的控制增益
	double _T = 0.0;//控制周期
public:
	KalmanFilterHelper() {

	}
private:

	void InitKalmanFilter(int xRows, int xCols)
	{
		T = 0.0;
		X = MatrixXd(xRows, xCols); X.setZero(); KFH_PRINT_BLOCK(X);
		_X = MatrixXd(X.rows(), X.cols());  _X.setZero(); KFH_PRINT_BLOCK(_X);
		F = MatrixXd(X.rows(), X.rows()); F.setZero(); KFH_PRINT_BLOCK(F);
		B = MatrixXd(F.rows(), X.cols()); B.setZero(); KFH_PRINT_BLOCK(B);
		H = MatrixXd(xRows / 2, X.rows()); H.setZero(); KFH_PRINT_BLOCK(H);
		Q = MatrixXd(F.rows(), F.cols()); Q.setZero(); KFH_PRINT_BLOCK(Q);
		R = MatrixXd(H.rows(), H.rows());  R.setZero(); KFH_PRINT_BLOCK(R);
		P = MatrixXd(F.rows(), F.cols());  P.setZero(); KFH_PRINT_BLOCK(P);
		K = MatrixXd(P.rows(), H.rows());  K.setZero(); KFH_PRINT_BLOCK(K);
		_Z = MatrixXd(H.rows(), X.cols()); _Z.setZero(); KFH_PRINT_BLOCK(_Z);
		I = MatrixXd::Identity(K.rows(), H.cols());  KFH_PRINT_BLOCK(I);
		_T = 0.0;
	}
	void InitKalmanFilter(int xRows, int xCols, int hRows)
	{
		T = 0.0;
		X = MatrixXd(xRows, xCols); X.setZero(); KFH_PRINT_BLOCK(X);
		_X = MatrixXd(X.rows(), X.cols());  _X.setZero(); KFH_PRINT_BLOCK(_X);
		F = MatrixXd(X.rows(), X.rows()); F.setZero(); KFH_PRINT_BLOCK(F);
		B = MatrixXd(F.rows(), X.cols()); B.setZero(); KFH_PRINT_BLOCK(B);
		H = MatrixXd(hRows, X.rows()); H.setZero(); KFH_PRINT_BLOCK(H);
		Q = MatrixXd(F.rows(), F.cols()); Q.setZero(); KFH_PRINT_BLOCK(Q);
		R = MatrixXd(H.rows(), H.rows());  R.setZero(); KFH_PRINT_BLOCK(R);
		P = MatrixXd(F.rows(), F.cols());  P.setZero(); KFH_PRINT_BLOCK(P);
		K = MatrixXd(P.rows(), H.rows());  K.setZero(); KFH_PRINT_BLOCK(K);
		_Z = MatrixXd(H.rows(), X.cols()); _Z.setZero(); KFH_PRINT_BLOCK(_Z);
		I = MatrixXd::Identity(K.rows(), H.cols());  KFH_PRINT_BLOCK(I);
		_T = 0.0;
	}
	void Predict()
	{
		X = F * _X + B * _U;
		P = F * P * F.transpose() + Q;
	}
	void Update()
	{
		K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
		X = X + K * (_Z - H * X);
		P = (I - K * H) * P;
	}
	/// <summary>
	/// 调用说明：
	///   调用PredictAndUpdate前，需设置_X、_Z、_U、_T参数
	/// </summary>
	/// <returns></returns>
	MatrixXd PredictAndUpdate()
	{
		//predict
		Predict();
		//update
		Update();
		return X;
	}
public:

	void InitKalmanFilter1D()
	{
		InitKalmanFilter(2, 1);
		X <<
			0, 0;
		F <<
			1, T,
			0, 1;
		B <<
			0.5 * T * T, T;
		H <<
			1, 0;
		Q <<
			0.01, 0,
			0, 0.01;
		R << 0.01;
		K <<
			0, 0;
	}
	MatrixXd kalmanFilter1D() {
		T = _T;
		F <<
			1, T,
			0, 1;
		B <<
			0.5 * T * T, T;

		return PredictAndUpdate();
	}
	void InitKalmanFilter2D() {
		InitKalmanFilter(4, 1);
	}
	MatrixXd kalmanFilter2D() {
		T = _T;
		F <<
			1, T,
			0, 1;
		B <<
			0.5 * T * T, T;
		return PredictAndUpdate();
	}
	void InitKalmanFilter3D() {
		InitKalmanFilter(6, 1);
	}
	MatrixXd kalmanFilter3D() {
		T = _T;
		F <<
			1, T,
			0, 1;
		B <<
			0.5 * T * T, T;
		return PredictAndUpdate();
	}
public:
	static KalmanFilterHelper* Inst() {
		static KalmanFilterHelper kalmanFilterHelperInstance;
		return &kalmanFilterHelperInstance;
	}
};

int main(int argc, char** argv)
{
	{
		KalmanFilterHelper::Inst()->InitKalmanFilter1D();
		std::cout << std::endl;
		KalmanFilterHelper::Inst()->InitKalmanFilter2D();
		std::cout << std::endl;
		KalmanFilterHelper::Inst()->InitKalmanFilter3D();
		return 0;
	}
	//""中是txt文件路径，注意：路径要用//隔开
	ofstream fout(".\\result.txt");

	double generateGaussianNoise(double mu, double sigma);//随机高斯分布数列生成器函数

	const double delta_t = 0.1;//控制周期，100ms
	const int num = 100;//迭代次数
	const double acc = 10;//加速度，ft/m

	MatrixXd A(2, 2);
	A(0, 0) = 1;
	A(1, 0) = 0;
	A(0, 1) = delta_t;
	A(1, 1) = 1;

	MatrixXd B(2, 1);
	B(0, 0) = pow(delta_t, 2) / 2;
	B(1, 0) = delta_t;

	MatrixXd H(1, 2);//测量的是小车的位移，速度为0
	H(0, 0) = 1;
	H(0, 1) = 0;

	MatrixXd Q(2, 2);//过程激励噪声协方差，假设系统的噪声向量只存在速度分量上，且速度噪声的方差是一个常量0.01，位移分量上的系统噪声为0
	Q(0, 0) = 0;
	Q(1, 0) = 0;
	Q(0, 1) = 0;
	Q(1, 1) = 0.01;

	MatrixXd R(1, 1);//观测噪声协方差，测量值只有位移，它的协方差矩阵大小是1*1，就是测量噪声的方差本身。
	R(0, 0) = 10;

	//time初始化，产生时间序列
	vector<double> time(100, 0);
	for (decltype(time.size()) i = 0; i != num; ++i) {
		time[i] = i * delta_t;
		//cout<<time[i]<<endl;
	}

	MatrixXd X_real(2, 1);
	vector<MatrixXd> x_real, rand;
	//生成高斯分布的随机数
	for (int i = 0; i < 100; ++i) {
		MatrixXd a(1, 1);
		a(0, 0) = generateGaussianNoise(0, sqrt(10));
		rand.push_back(a);
	}
	//生成真实的位移值
	for (int i = 0; i < num; ++i) {
		X_real(0, 0) = 0.5 * acc * pow(time[i], 2);
		X_real(1, 0) = 0;
		x_real.push_back(X_real);
	}

	//变量定义，包括状态预测值，状态估计值，测量值，预测状态与真实状态的协方差矩阵，估计状态和真实状态的协方差矩阵，初始值均为零
	MatrixXd X_evlt = MatrixXd::Constant(2, 1, 0), X_pdct = MatrixXd::Constant(2, 1, 0), Z_meas = MatrixXd::Constant(1, 1, 0),
		Pk = MatrixXd::Constant(2, 2, 0), Pk_p = MatrixXd::Constant(2, 2, 0), K = MatrixXd::Constant(2, 1, 0);
	vector<MatrixXd> x_evlt, x_pdct, z_meas, pk, pk_p, k;
	x_evlt.push_back(X_evlt);
	x_pdct.push_back(X_pdct);
	z_meas.push_back(Z_meas);
	pk.push_back(Pk);
	pk_p.push_back(Pk_p);
	k.push_back(K);

	//开始迭代
	for (int i = 1; i < num; ++i) {
		//预测值
		X_pdct = A * x_evlt[i - 1] + B * acc;
		x_pdct.push_back(X_pdct);
		//预测状态与真实状态的协方差矩阵，Pk'
		Pk_p = A * pk[i - 1] * A.transpose() + Q;
		pk_p.push_back(Pk_p);
		//K:2x1
		MatrixXd tmp(1, 1);
		auto hph_t = H * pk_p[i] * H.transpose();
		tmp = H * pk_p[i] * H.transpose() + R;
		auto tmp_inv = tmp.inverse();
		K = pk_p[i] * H.transpose() * tmp.inverse();
		k.push_back(K);
		//测量值z
		Z_meas = H * x_real[i] + rand[i];
		z_meas.push_back(Z_meas);
		//估计值
		X_evlt = x_pdct[i] + k[i] * (z_meas[i] - H * x_pdct[i]);
		x_evlt.push_back(X_evlt);
		//估计状态和真实状态的协方差矩阵，Pk
		Pk = (MatrixXd::Identity(2, 2) - k[i] * H) * pk_p[i];
		pk.push_back(Pk);
	}

	cout << "含噪声测量" << "  " << "后验估计" << "  " << "真值" << "  " << endl;
	for (int i = 0; i < num; ++i) {
		cout << z_meas[i] << "  " << x_evlt[i](0, 0) << "  " << x_real[i](0, 0) << endl;
		fout << z_meas[i] << "  " << x_evlt[i](0, 0) << "  " << x_real[i](0, 0) << endl;//输出到txt文档，用于matlab绘图
																						//cout<<k[i](1,0)<<endl;
																						//fout<<rand[i](0,0)<<endl;
																						//fout<<x_pdct[i](0,0)<<endl;
	}

	fout.close();
	getchar();
	return 0;
}

//生成高斯分布随机数的函数，网上找的
double generateGaussianNoise(double mu, double sigma)
{
	const double epsilon = std::numeric_limits<double>::min();
	const double two_pi = 2.0 * 3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
		return z1 * sigma + mu;

	double u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}