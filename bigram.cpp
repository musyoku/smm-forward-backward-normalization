# include <iostream>
# include <random>
using std::cout;
using std::endl;

double enumerate_forward_probability_naive(double*** p_transition, double** alpha, int seq_length, int max_word_length){
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
	return px;
}

double enumerate_forward_probability_logsumexp(double*** p_transition, double** alpha, double* log_z, int seq_length, int max_word_length){
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

		}
		double sum_exp = 0;
		// 最大値を求める
		double max_log_z = 0;

		if(t - k == 0){
			double tmp = log(alpha[t][k][0]) + _log_z[t - k];
			if(max_log_z == 0 || tmp > max_log_z){
				max_log_z = tmp;
			}
		}
		for(int j = 1;j <= std::min(t - k, _max_word_length);j++){
			double tmp = log(alpha[t][k][j]) + _log_z[t - k];
			if(max_log_z == 0 || tmp > max_log_z){
				max_log_z = tmp;
			}
		}
	}
	// <eos>への遷移を考える
	double px = 0;
	int t = seq_length;
	for(int k = 1;k <= std::min(seq_length, max_word_length);k++){
		px += alpha[t][k] * p_transition[t + 1][1][k];
	}
	return px;
}

// tは番号なので1から始まることに注意
int main(int argc, char *argv[]){
	int seq_length = 10;
	int max_word_length = 10;
	// 前向き確率
	double** alpha = new double*[seq_length + 1];
	for(int t = 0;t < seq_length + 1;t++){
		alpha[t] = new double[max_word_length + 1];
	}
	// 後向き確率
	double** beta = new double*[seq_length + 1];
	for(int t = 0;t < seq_length + 1;t++){
		beta[t] = new double[max_word_length + 1];
	}
	// 遷移確率をランダムに決める
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	double*** p_transition = new double**[seq_length + 2];	// 最後に<eos>への遷移確率を入れる
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

	double px_true = enumerate_forward_probability_naive(p_transition, alpha, seq_length, max_word_length);
	cout << px_true << endl;

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
}