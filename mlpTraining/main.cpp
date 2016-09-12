#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

void get_data(string filename, int num_features, int &num_samples, Mat &_data, Mat &_response);
inline TermCriteria TC(int iters, double eps);

int main(int argc, char **argv)
{
	int num_samples;
	int class_count = 3;
	Mat data;
	Mat response;
	get_data("Results.txt",48,num_samples,data, response);

	Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, response);

	Ptr<ANN_MLP> model;

	int layer_sz[] = { data.cols, 100, 100, class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.001;
	int max_iter = 300;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	//Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

	cout << "Training the classifier (may take a few minutes)...\n";
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TC(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	//model->save("test1");
	cout << endl;
	Mat sample = data.row(32);
	float r = model->predict(sample);
	cout << r << endl;
}

void get_data(string filename, int num_features, int &num_samples, Mat &_data, Mat &_response)
{
	ifstream file(filename);
	string str;
	//int num_samples = 0;
	//int num_features = 48;
	int class_count = 3;

	int class_entry;
	float data_entry;

	Mat response(0, class_count, CV_32F);
	Mat t_data(0, num_features, CV_32F);

	if (!file)
		cout << "Error opening file";

	while (getline(file, str))
	{
		istringstream temp(str);
		Mat data_row(1, num_features, CV_32F);
		Mat response_row = Mat::zeros(1, class_count, CV_32F);

		temp >> class_entry;
		response_row.at<float>(0, class_entry) = 1.f;
		response.push_back(response_row);
		//response.push_back((float)class_entry);

		for (int i = 0; i < num_features; ++i)
		{
			temp >> data_entry;
			data_row.at<float>(0, i++) = data_entry;
		}

		t_data.push_back(data_row);
	}
	num_samples = t_data.rows;

	Mat(response).copyTo(_response);
	Mat(t_data).copyTo(_data);
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}