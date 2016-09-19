#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <array>

using namespace std;
using namespace cv;
using namespace cv::ml;

bool get_data(string filename, int num_features, int &num_samples, Mat &_data, Mat &_response, Mat &_class_response);
inline TermCriteria TC(int iters, double eps);
bool trainMLP(string dataPath, string modelPath, string nodeconfig);
bool predictMLP(string dataPath, string modelPath, string resFilename);
//int * getnodeconfig(int input_nodes, string nodeconfig, int output_nodes, int &num_layers);
vector<int> getnodeconfig(int input_nodes, string nodeconfig, int output_nodes);
void calcResults(Mat pred, Mat resp, string resFilename);

struct pmetrics
{
	float FPrate;
	float TPrate;
	float Precision;
	float Fscore;
	float Accuracy;
};


int main(int argc, char **argv)
{
	//command line options and default values
	const String keys =
		"{help h usage ?|| print this message}"
		"{train         || training mode				}"
		"{modelfile     | model.xml | path to model file        }"
		"{datafile      | test.txt | path to data file         }"
		"{nodeconfig    | 2 100 100 | node configuration         }"
		"{resfilename   | predout.txt| results of prediction}"
		;

	bool train = false;

	CommandLineParser parser(argc, argv, keys); //paser object

	if (parser.has("help"))
	{
		cout << "The usage: mlpTraining.exe [-datafile=<path to training / prediction data>] \\\n"
			"  [-modelfile=<file for the classifier>] \\\n"
			"  [-nodeconfig=\"[number of hidden layers] [nodes in hidden layer 1] [nodes in hidden layer 2] [..]..\"] \\\n"
			"  [-train] # to train. omit for prediction \\\n"
			"  [-resfilename=<filename to store results> \\\n" << endl;
		return 0;
	}

	if (parser.has("train"))	//check if training or prediction mode
	{
		train = true;
	}
	string modelpath = parser.get<string>("modelfile");		//get name of mdel file
	string datapath = parser.get<string>("datafile");		//get name of data file
	string nodeconfig = parser.get<string>("nodeconfig");	// get node configuration
	string resfilename = parser.get<string>("resfilename");	// get node configuration

	if (train)	//training mode
	{
		cout << "Training mode" << endl;
		double start = (double)getTickCount();
		if (trainMLP(datapath, modelpath, nodeconfig))		//call training function
		{
			double end = (double)getTickCount();
			double  trainTime = (end - start) / (getTickFrequency());
			cout << "Model training took " << trainTime << " seconds" << endl;	//show elapsed time
			cout << "Model training successful" << endl;
		}
		else
			cout << "Model training failed" << endl;

	}
	else
	{
		cout << "Predicton mode" << endl;
		if (predictMLP(datapath, modelpath, resfilename))				//call prediction function
			cout << "Model prediction successful" << endl;
	}

	return 0;
}

// read features from file
bool get_data(string filename, int num_features, int &num_samples, Mat &_data, Mat &_response, Mat &_class_response)
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
	Mat class_response(0, 1, CV_32F);

	if (!file)
	{
		cout << "Error opening file";
		return false;
	}

	while (getline(file, str))
	{
		istringstream temp(str);
		Mat data_row(1, num_features, CV_32F);
		Mat response_row = Mat::zeros(1, class_count, CV_32F);

		temp >> class_entry;
		response_row.at<float>(0, class_entry) = 1.f;
		response.push_back(response_row);
		class_response.push_back((float)class_entry);

		for (int i = 0; i < num_features; ++i)
		{
			temp >> data_entry;
			data_row.at<float>(0, i) = data_entry;
		}

		t_data.push_back(data_row);
	}
	num_samples = t_data.rows;

	Mat(response).copyTo(_response);
	Mat(t_data).copyTo(_data);
	Mat(class_response).copyTo(_class_response);

	return true;
}

//parse the node configuration
vector<int> getnodeconfig(int input_nodes, string nodeconfig, int output_nodes)
{
	vector<int> layer_nodes;

	layer_nodes.push_back(input_nodes);

	istringstream ss(nodeconfig);
	int inner_nodes;
	ss >> inner_nodes;

	for (int i = 1; i < inner_nodes + 2 - 1; ++i)
	{
		int num_nodes;
		ss >> num_nodes;
		layer_nodes.push_back(num_nodes);
	}

	layer_nodes.push_back(output_nodes);

	return layer_nodes;
}

//int * getnodeconfig(int input_nodes, string nodeconfig, int output_nodes, int &num_layers)
//{
//	istringstream ss(nodeconfig);
//	int inner_nodes;
//	ss >> inner_nodes;
//	int *nodes = new int[inner_nodes + 2];
//
//	for (int i = 1; i < inner_nodes + 2 - 1; ++i)
//		ss >> nodes[i];
//
//	nodes[0] = input_nodes;
//	nodes[inner_nodes + 2 - 1] = output_nodes;
//
//	num_layers = inner_nodes + 2;
//
//	return nodes;
//}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

bool trainMLP(string dataPath, string modelPath, string nodeconfig)
{
	int num_samples;
	int class_count = 3;
	Mat data;
	Mat response;
	Mat class_response;
	if (!get_data(dataPath, 48, num_samples, data, response, class_response))
		return false;

	cout << "Number of samples for training: " << num_samples << endl;

	Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, response);

	Ptr<ANN_MLP> model;

	//int layer_sz[] = { data.cols, 100, 100, class_count };
	//int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	//int nlayers;
	//int * layer_sz = getnodeconfig(data.cols, nodeconfig, class_count, nlayers);
	//Mat layer_sizes(1, nlayers, CV_32S, layer_sz);
	vector<int> layer_sizes = getnodeconfig(data.cols, nodeconfig, class_count);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.001;
	int max_iter = 1000;//300;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	cout << "Training the classifier (may take a few minutes)...\n";
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TC(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	model->save(modelPath);

	return true;
}

bool predictMLP(string dataPath, string modelPath, string resFilename)
{
	int num_samples;
	int class_count = 3;
	Mat data;
	Mat response;
	Mat class_response;
	if (!get_data(dataPath, 48, num_samples, data, response, class_response))
		return false;

	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>(modelPath);

	if (model.empty())
		return false;

	Mat pred_response(0, 1, CV_32F);

	for (int i = 0; i < data.rows; ++i)
		pred_response.push_back(model->predict(data.row(i)));

	calcResults(pred_response, class_response, resFilename);

	return true;
}

void calcResults(Mat pred, Mat resp, string resFilename)
{
	vector<int> correct;
	vector<int> true_responses;
	vector<int> predicted_responses;
	array<int, 3> TruePositive = { 0 };
	array<int, 3> FalsePositive = { 0 };
	array<int, 3> TrueNegative = { 0 };
	array<int, 3> FalseNegative = { 0 };
	array<array<int, 3>, 3> matrix = { 0 };
	//array<float, 3> FPrate = { 0 };
	//array<float, 3> TPrate = { 0 };
	//array<float, 3> Precision = { 0 };
	//array<float, 3> Accuracy = { 0 };
	//array<float, 3> Fscore = { 0 };
	array<pmetrics, 3> metrics;

	for(int i=0;i<pred.rows; ++i)
	{
		correct.push_back(std::abs(pred.at<float>(i) - resp.at<float>(i)) <= FLT_EPSILON ? 1 : 0);
	}

	for (int i = 0; i < resp.rows; ++i)
	{
		cout << resp.at<float>(i, 0) << " " << pred.at<float>(i, 0) <<" " << correct[i] << endl;
	}

	cout << endl;

	for (int i = 0; i < resp.rows; ++i)
	{
		if (abs(resp.at<float>(i, 0) - 0) < 0.1) //class 0
		{
			true_responses.push_back(0);
		}
		else if (abs(resp.at<float>(i, 0) - 1) < 0.1)
		{
			true_responses.push_back(1);
		}
		else if (abs(resp.at<float>(i, 0) - 2) < 0.1)
		{
			true_responses.push_back(2);
		}
		else
		{
			cout << "Error" << endl;
		}
	}

	for (int i = 0; i < pred.rows; ++i)
	{
		if (abs(pred.at<float>(i, 0) - 0) < 0.1) //class 0
		{
			predicted_responses.push_back(0);
		}
		else if (abs(pred.at<float>(i, 0) - 1) < 0.1)
		{
			predicted_responses.push_back(1);
		}
		else if (abs(pred.at<float>(i, 0) - 2) < 0.1)
		{
			predicted_responses.push_back(2);
		}
		else
		{
			cout << "Error" << endl;
		}
	}

	for (int i = 0; i < true_responses.size(); i++)
	{
		matrix[true_responses[i]][predicted_responses[i]]++;
	}

	for (int i = 0; i < true_responses.size(); i++)
	{
		if (true_responses[i] == 0)
		{
			if (predicted_responses[i] == 0)
			{
				TruePositive[0]++;
				TrueNegative[1]++;
				TrueNegative[2]++;
			}
			else if (predicted_responses[i] == 1)
			{
				FalseNegative[0]++;
				FalsePositive[1]++;
			}
			else if (predicted_responses[i] == 2)
			{
				FalseNegative[0]++;
				FalsePositive[2]++;
			}
		}
		else if (true_responses[i] == 1)
		{
			if (predicted_responses[i] == 1)
			{
				TruePositive[1]++;
				TrueNegative[0]++;
				TrueNegative[2]++;
			}
			else if (predicted_responses[i] == 0)
			{
				FalseNegative[1]++;
				FalsePositive[0]++;
			}
			else if (predicted_responses[i] == 2)
			{
				FalseNegative[1]++;
				FalsePositive[2]++;
			}
		}
		else if (true_responses[i] == 2)
		{
			if (predicted_responses[i] == 2)
			{
				TruePositive[2]++;
				TrueNegative[0]++;
				TrueNegative[1]++;
			}
			else if (predicted_responses[i] == 0)
			{
				FalseNegative[2]++;
				FalsePositive[0]++;
			}
			else if (predicted_responses[i] == 1)
			{
				FalseNegative[2]++;
				FalsePositive[1]++;
			}
		}

	}

	for (int i = 0; i < 3; ++i)
	{
		metrics[i].FPrate = FalsePositive[i] / ((float)(FalsePositive[i] + TrueNegative[i]));
		metrics[i].TPrate = TruePositive[i] / ((float)(TruePositive[i] + FalseNegative[i]));
		metrics[i].Precision = TruePositive[i] / ((float)(TruePositive[i] + FalsePositive[i]));
		metrics[i].Fscore = metrics[i].Precision * metrics[i].TPrate;
		metrics[i].Accuracy = (TruePositive[i] + TrueNegative[i]) / ((float)(TruePositive[i] + FalsePositive[i] + FalseNegative[i] + TrueNegative[i]));
	}

	
	for (int i = 0; i < 3; ++i)
	{
		//cout << i << " " << "TP: " << TruePositive[i] << " FP: " << FalsePositive[i] << " TN: " << TrueNegative[i] << " FN: " << FalseNegative[i] << endl;
		//cout << i << " " << "TP: " << TruePositive[i] << " FP: " << FalsePositive[i] << " TN: " << TrueNegative[i] << " FN: " << FalseNegative[i] << endl;
		cout << i << ":" << endl;
		cout << endl;
		cout << TruePositive[i] << "\t" << FalsePositive[i] << endl;
		cout << FalseNegative[i] << "\t" << TrueNegative[i] << endl;
		cout << endl;

		cout << "FP rate: " << metrics[i].FPrate << endl;
		cout << "TP rate: " << metrics[i].TPrate << endl;
		cout << "Precision: " << metrics[i].Precision << endl;
		cout << "F-score: " << metrics[i].Fscore << endl;
		cout << "Accuracy: " << metrics[i].Accuracy << endl;
		cout << endl;
	}

	ofstream outfile(resFilename);

	for (int i = 0; i < 3; ++i)
	{
		outfile << i << ":" << endl;
		outfile << endl;
		outfile << TruePositive[i] << "\t" << FalsePositive[i] << endl;
		outfile << FalseNegative[i] << "\t" << TrueNegative[i] << endl;
		outfile << endl;

		outfile << "FP rate: " << metrics[i].FPrate << endl;
		outfile << "TP rate: " << metrics[i].TPrate << endl;
		outfile << "Precision: " << metrics[i].Precision << endl;
		outfile << "F-score: " << metrics[i].Fscore << endl;
		outfile << "Accuracy: " << metrics[i].Accuracy << endl;
		outfile << endl;
	}


	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}

}