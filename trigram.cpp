#include <iostream>
#include <random>
#include <cassert>
#include <chrono>
using std::cout;
using std::endl;

double enumerate_forward_probability_naive(double**** p_transition, double*** alpha, int seq_length, int max_word_length){
	for(int t = 1;t <= seq_length;t++){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] = 0;
			if(t - k == 0){
				alpha[t][k] = p_transition[t][k][0];
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				alpha[t][k] += p_transition[t][k][j] * alpha[t - k][j];
			}
		}
	}
	// <eos>への遷移を考える
	double px = 0;
	int t = seq_length;
	for(int k = 1;k <= std::min(seq_length, max_word_length);k++){
		px += alpha[t][k] * p_transition[t + 1][1][k];
	}
	return log(px);
}

double enumerate_forward_probability_logsumexp(double**** p_transition, double*** alpha, double* log_z, int seq_length, int max_word_length){
	for(int t = 1;t <= seq_length;t++){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] = 0;
			if(t - k == 0){
				alpha[t][k] = p_transition[t][k][0];
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				alpha[t][k] += p_transition[t][k][j] * alpha[t - k][j];
			}
		}
		if(t == 1){
			log_z[t] = log(alpha[t][1]);
			alpha[t][1] = 1;
			continue;
		}
		// 最大値を求める
		double max_log_z = 0;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			double tmp = log(alpha[t][k]) + log_z[t - k];
			if(max_log_z == 0 || tmp > max_log_z){
				max_log_z = tmp;
			}
		}
		// 求めた最大値をもとにlogsumexp
		double sum_exp = 0;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			sum_exp += exp(log(alpha[t][k]) + log_z[t - k] - max_log_z);
		}
		double log_z_t = log(sum_exp) + max_log_z;
		// 正規化
		assert(log_z_t != 0);
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] = exp(log(alpha[t][k]) + log_z[t - k] - log_z_t);
		}
		log_z[t] = log_z_t;
		continue;
	}
	// <eos>への遷移を考える
	double alpha_t_1 = 0;
	int t = seq_length + 1;
	for(int k = 1;k <= std::min(t, max_word_length);k++){
		alpha_t_1 += alpha[t - 1][k] * p_transition[t][1][k];
	}
	double log_z_t = log(alpha_t_1) + log_z[t - 1];
	// return log_z_t;	// 実はこれを返してもよい。なぜなら正規化定数は前向き確率そのもの
	alpha_t_1 = exp(log(alpha_t_1) + log_z[t - 1] - log_z_t);	// 正規化
	return log(alpha_t_1) + log_z_t;							// 正規化を元に戻す
}

double enumerate_forward_probability_scaling(double**** p_transition, double*** alpha, double* scaling, int seq_length, int max_word_length){
	for(int t = 1;t <= seq_length;t++){
		// cout << "t = " << t << endl;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] = 0;
			double prod_scaling = 1;
			for(int m = t - k + 1;m <= t - 1;m++){
				prod_scaling *= scaling[m];
			}
			if(t - k == 0){
				alpha[t][k] = p_transition[t][k][0] * prod_scaling;
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				alpha[t][k] += p_transition[t][k][j] * alpha[t - k][j] * prod_scaling;
			}
		}
		double sum_alpha = 0;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			sum_alpha += alpha[t][k];
		}
		// assert(sum_alpha <= 1.0);
		scaling[t] = 1.0 / sum_alpha;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] *= scaling[t];
		}
	}
	// <eos>への遷移を考える
	double alpha_t_1 = 0;
	int t = seq_length + 1;
	for(int j = 1;j <= std::min(t, max_word_length);j++){
		alpha_t_1 += alpha[t - 1][j] * p_transition[t][1][j];
	}
	scaling[t] = 1.0 / alpha_t_1;
	double log_px = 0;
	for(int m = 1;m <= t;m++){
		log_px += log(1.0 / scaling[m]);
	}
	return log_px;
}

// 高速版
double _enumerate_forward_probability_scaling(double**** p_transition, double*** alpha, double* scaling, int seq_length, int max_word_length){
	for(int t = 1;t <= seq_length;t++){
		// cout << "t = " << t << endl;
		double prod_scaling = 1;
		double sum_alpha = 0;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] = 0;
			if(k > 1){
				prod_scaling *= scaling[t - k + 1];
			}
			if(t - k == 0){
				alpha[t][k] = p_transition[t][k][0] * prod_scaling;
				sum_alpha += alpha[t][k];
				continue;
			}
			for(int j = 1;j <= std::min(t - k, max_word_length);j++){
				alpha[t][k] += p_transition[t][k][j] * alpha[t - k][j] * prod_scaling;
			}
			sum_alpha += alpha[t][k];
		}
		// assert(sum_alpha <= 1.0);
		scaling[t] = 1.0 / sum_alpha;
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			alpha[t][k] *= scaling[t];
		}
	}
	// <eos>への遷移を考える
	double alpha_t_1 = 0;
	int t = seq_length + 1;
	for(int j = 1;j <= std::min(t, max_word_length);j++){
		alpha_t_1 += alpha[t - 1][j] * p_transition[t][1][j];
	}
	scaling[t] = 1.0 / alpha_t_1;
	double log_px = 0;
	for(int m = 1;m <= t;m++){
		log_px += log(1.0 / scaling[m]);
	}
	return log_px;
}

// tは番号なので1から始まることに注意
int main(int argc, char *argv[]){
	int seq_length = 200;
	int max_word_length = 10;
	// 前向き確率
	double*** alpha = new double**[seq_length + 1];
	for(int t = 0;t < seq_length + 1;t++){
		alpha[t] = new double*[max_word_length + 1];
		for(int k = 0;k <= max_word_length;k++){
			alpha[t][k] = new double[max_word_length + 1];
		}
	}
	// 後向き確率
	double** beta = new double*[seq_length + 1];
	for(int t = 0;t < seq_length + 1;t++){
		beta[t] = new double[max_word_length + 1];
		for(int k = 0;k <= max_word_length;k++){
			alpha[t][k] = new double[max_word_length + 1];
		}
	}
	// 遷移確率をランダムに決める
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	double**** p_transition = new double**[seq_length + 2];	// 最後に<eos>への遷移確率を入れる
	for(int t = 0;t < seq_length + 2;t++){
		p_transition[t] = new double*[max_word_length + 1];
		for(int k = 0;k < max_word_length + 1;k++){
			p_transition[t][k] = new double[max_word_length + 1];
			for(int j = 0;j < max_word_length + 1;j++){
				p_transition[t][k][j] = uniform(mt) / 1000.0;
			}
		}
	}
	// 正規化定数（logsumexp用）
	double* log_z = new double[seq_length + 1];
	// スケーリング係数
	double* scaling = new double[seq_length + 2];

	int repeat = 1000;
	double log_px_true, log_px_logsumexp, log_px_scaling, _log_px_scaling;

    auto start = std::chrono::system_clock::now();
    for(int r = 0;r < repeat;r++){
		log_px_true = enumerate_forward_probability_naive(p_transition, alpha, seq_length, max_word_length);
    }
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    cout << "naive:		" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

    start = std::chrono::system_clock::now();
    for(int r = 0;r < repeat;r++){
		log_px_logsumexp = enumerate_forward_probability_logsumexp(p_transition, alpha, log_z, seq_length, max_word_length);
	}
    end = std::chrono::system_clock::now();
    diff = end - start;
    cout << "logsumexp:	" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

    start = std::chrono::system_clock::now();
    for(int r = 0;r < repeat;r++){
		log_px_scaling = enumerate_forward_probability_scaling(p_transition, alpha, scaling, seq_length, max_word_length);
	}
    end = std::chrono::system_clock::now();
    diff = end - start;
    cout << "scaling:	" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

    start = std::chrono::system_clock::now();
    for(int r = 0;r < repeat;r++){
		_log_px_scaling = _enumerate_forward_probability_scaling(p_transition, alpha, scaling, seq_length, max_word_length);
	}
    end = std::chrono::system_clock::now();
    diff = end - start;
    cout << "_scaling:	" << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / (double)repeat) << " [msec]" << endl;

	cout << log_px_true << endl;
	cout << log_px_logsumexp << endl;
	cout << log_px_scaling << endl;
	cout << _log_px_scaling << endl;

	for(int t = 0;t < seq_length + 1;t++){
		for(int k = 0;k < max_word_length + 1;k++){
			delete[] p_transition[t][k];
		}
		delete[] p_transition[t];
		delete[] alpha[t];
		delete[] beta[t];
	}
	delete[] p_transition;
	delete[] alpha;
	delete[] beta;
	delete[] log_z;
	delete[] scaling;
}