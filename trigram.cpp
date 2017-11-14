#include <iostream>
#include <random>
#include <cassert>
#include <chrono>
#include <iomanip>
using std::cout;
using std::endl;

double enumerate_forward_probability_naive(double**** p_transition, double*** alpha, int seq_length, int max_word_length){
	for(int t = 1;t <= seq_length;t++){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			if(t - k == 0){
				alpha[t][k][0] = p_transition[t][k][0][0];
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				if(t - k - j == 0){
					alpha[t][k][j] = p_transition[t][k][j][0] * alpha[t - k][j][0];
					continue;
				}
				alpha[t][k][j] = 0;
				for(int i = 1;i <= std::min(t - k, max_word_length);i++){
					alpha[t][k][j] += p_transition[t][k][j][i] * alpha[t - k][j][i];
				}
			}
		}
	}
	// <eos>への遷移を考える
	double px = 0;
	int t = seq_length + 1;
	for(int j = 1;j <= std::min(seq_length, max_word_length);j++){
		for(int i = 1;i <= std::min(seq_length - j, max_word_length);i++){
			px += alpha[t - 1][j][i] * p_transition[t][1][j][i];
		}
	}
	return log(px);
}

double enumerate_backward_probability_naive(double**** p_transition, double*** beta, int seq_length, int max_word_length){
	// <eos>への遷移を考える
	int t = seq_length;
	for(int j = 1;j <= std::min(seq_length, max_word_length);j++){
		for(int i = 1;i <= std::min(seq_length - j, max_word_length);i++){
			beta[t][j][i] = p_transition[t + 1][1][j][i];
		}
	}
	for(int t = seq_length - 1;t >= 1;t--){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			for(int j = 0;j <= std::min(t - k, max_word_length);j++){
				beta[t][k][j] = 0;
				for(int i = 1;i <= std::min(seq_length - t, max_word_length);i++){
					beta[t][k][j] += p_transition[t + i][i][k][j] * beta[t + i][i][k];
				}
				assert(0 < beta[t][k][j] && beta[t][k][j] <= 1.0);
			}
		}
	}
	double px = 0;
	t = 0;
	for(int j = 1;j <= std::min(seq_length, max_word_length);j++){
		px += p_transition[j][j][0][0] * beta[j][j][0];
	}
	return log(px);
}

double enumerate_forward_probability_logsumexp(double**** p_transition, double*** alpha, double* log_z, int seq_length, int max_word_length){
	alpha[0][0][0] = 1;
	log_z[0] = 0;
	for(int t = 1;t <= seq_length;t++){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			if(t - k == 0){
				alpha[t][k][0] = p_transition[t][k][0][0];
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				if(t - k - j == 0){
					alpha[t][k][j] = p_transition[t][k][j][0] * alpha[t - k][j][0];
					continue;
				}
				alpha[t][k][j] = 0;
				for(int i = 1;i <= std::min(t - k, max_word_length);i++){
					alpha[t][k][j] += p_transition[t][k][j][i] * alpha[t - k][j][i];
				}
			}
		}
		// 正規化
		// 分配関数はkとjを網羅する
		double sum_exp = 0;
		// 最大値を求める
		double max_log_z = 0;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			if(t - k == 0){
				assert(alpha[t][k][0] > 0);
				double tmp = log(alpha[t][k][0]) + log_z[t - k];
				if(max_log_z == 0 || tmp > max_log_z){
					max_log_z = tmp;
				}
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				assert(alpha[t][k][j] > 0);
				double tmp = log(alpha[t][k][j]) + log_z[t - k];
				if(max_log_z == 0 || tmp > max_log_z){
					max_log_z = tmp;
				}
			}
		}
		// 求めた最大値をもとにlogsumexp
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			if(t - k == 0){
				sum_exp += exp(log(alpha[t][k][0]) + log_z[t - k] - max_log_z);
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				assert(alpha[t][k][j] > 0);
				sum_exp += exp(log(alpha[t][k][j]) + log_z[t - k] - max_log_z);
			}
		}
		double log_z_t = log(sum_exp) + max_log_z;
		// 正規化
		assert(log_z_t != 0);
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			if(t - k == 0){
				alpha[t][k][0] = exp(log(alpha[t][k][0]) + log_z[t - k] - log_z_t);
				assert(alpha[t][k][0] > 0);
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				alpha[t][k][j] = exp(log(alpha[t][k][j]) + log_z[t - k] - log_z_t);
				assert(alpha[t][k][j] > 0);
			}
		}
		log_z[t] = log_z_t;
	}
	double alpha_t_1 = 0;
	int t = seq_length + 1;
	for(int j = 1;j <= std::min(seq_length, max_word_length);j++){
		for(int i = 1;i <= std::min(seq_length - j, max_word_length);i++){
			alpha_t_1 += p_transition[t][1][j][i] * alpha[t - 1][j][i];
		}
	}
	return log(alpha_t_1) + log_z[t - 1];
}

void init_log_z(double* log_z, int seq_length){
	for(int t = 0;t <= seq_length;t++){
		log_z[t] = 0;
	}
}

// tは番号なので1から始まることに注意
int main(int argc, char *argv[]){
	int seq_length = 1000;
	int max_word_length = 10;
	// 前向き確率
	double*** alpha = new double**[seq_length + 1];
	for(int t = 0;t <= seq_length;t++){
		alpha[t] = new double*[max_word_length + 1];
		for(int k = 0;k <= max_word_length;k++){
			alpha[t][k] = new double[max_word_length + 1];
		}
	}
	// 後向き確率
	double*** beta = new double**[seq_length + 2];
	for(int t = 0;t <= seq_length + 1;t++){
		beta[t] = new double*[max_word_length + 1];
		for(int k = 0;k <= max_word_length;k++){
			beta[t][k] = new double[max_word_length + 1];
		}
	}
	// 遷移確率をランダムに決める
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	double**** p_transition = new double***[seq_length + 2];	// 最後に<eos>への遷移確率を入れる
	for(int t = 0;t <= seq_length + 1;t++){
		p_transition[t] = new double**[max_word_length + 1];
		for(int k = 0;k <= max_word_length;k++){
			p_transition[t][k] = new double*[max_word_length + 1];
			for(int j = 0;j <= max_word_length;j++){
				p_transition[t][k][j] = new double[max_word_length + 1];
				for(int i = 0;i <= max_word_length;i++){
					p_transition[t][k][j][i] = uniform(mt) / 1000.0;
				}
			}
		}
	}
	// 正規化定数（logsumexp用）
	double* log_z = new double[seq_length + 2];
	// スケーリング係数
	double* scaling = new double[seq_length + 2];
	// log,expの計算結果の保存用
	double* log_exp_cache = new double[max_word_length + 1];

	int repeat = 1000;
	double log_px_true_forward, log_px_logsumexp_forward, _log_px_logsumexp_forward, log_px_scaling_forward, _log_px_scaling_forward;
	double log_px_true_backward, log_px_logsumexp_backward, _log_px_logsumexp_backward, log_px_scaling_backward, _log_px_scaling_backward;

	cout << "trigram:" << endl;
	cout << "forward variables:" << endl;
	cout << "	time:" << endl;

	auto start = std::chrono::system_clock::now();
	for(int r = 0;r < repeat;r++){
		log_px_true_forward = enumerate_forward_probability_naive(p_transition, alpha, seq_length, max_word_length);
	}
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	cout << "		naive:			" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

	start = std::chrono::system_clock::now();
	for(int r = 0;r < repeat;r++){
		log_px_logsumexp_forward = enumerate_forward_probability_logsumexp(p_transition, alpha, log_z, seq_length, max_word_length);
	}
	end = std::chrono::system_clock::now();
	diff = end - start;
	cout << "		logsumexp:		" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

	cout << "	logP(x):" << endl;
	cout << "		" << std::setprecision(16) << log_px_true_forward << endl;
	cout << "		" << std::setprecision(16) << log_px_logsumexp_forward << endl;
	cout << "		" << std::setprecision(16) << _log_px_logsumexp_forward << endl;
	// cout << "		" << std::setprecision(16) << log_px_scaling_forward << endl;
	cout << "		" << std::setprecision(16) << _log_px_scaling_forward << endl;

	cout << "backward variables:" << endl;
	cout << "	time:" << endl;

	start = std::chrono::system_clock::now();
	for(int r = 0;r < repeat;r++){
		log_px_true_backward = enumerate_backward_probability_naive(p_transition, beta, seq_length, max_word_length);
	}
	end = std::chrono::system_clock::now();
	diff = end - start;
	cout << "		naive:			" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

	cout << "	logP(x):" << endl;
	cout << "		" << std::setprecision(16) << log_px_true_backward << endl;
	// cout << "		" << std::setprecision(16) << log_px_logsumexp_backward << endl;
	cout << "		" << std::setprecision(16) << _log_px_logsumexp_backward << endl;
	// cout << "		" << std::setprecision(16) << log_px_scaling_backward << endl;
	cout << "		" << std::setprecision(16) << _log_px_scaling_backward << endl;


	for(int t = 0;t <= seq_length;t++){
		delete[] alpha[t];
	}
	for(int t = 0;t <= seq_length + 1;t++){
		for(int k = 0;k <= max_word_length;k++){
			for(int j = 0;j <= max_word_length;j++){
				delete[] p_transition[t][k][j];
			}
			delete[] p_transition[t][k];
		}
		delete[] p_transition[t];
		delete[] beta[t];
	}
	delete[] p_transition;
	delete[] alpha;
	delete[] beta;
	delete[] log_z;
	delete[] scaling;
	delete[] log_exp_cache;
}