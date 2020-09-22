#include <iostream>
#include "fsrcnnTrainer.h"
#include "fsrcnnNet.h"
#include <torch/script.h>

cv::Mat test_pic(const cv::Mat& img)
{
	auto net = torch::jit::load("../model/epoch_279.pt");
	net.eval();

	if(torch::cuda::is_available())
		net.to(torch::kCUDA);
	std::vector<cv::Mat> ycrcb;
	cv::split(img, ycrcb);

	cv::Mat y = *ycrcb.begin();
	torch::Tensor tensor = torch::from_blob(y.data, {1, y.rows, y.cols, 1}, torch::kByte);
	tensor = tensor.permute({0,3,1,2});
	tensor = tensor.toType(torch::kFloat).div(255);

	if(torch::cuda::is_available())
		tensor = tensor.cuda();
	auto res = net.forward({tensor}).toTensor();

	cv::Mat y_res = Tensor_y_Tocv8uc1(res);
	ycrcb.front() = y_res.clone();
	cv::resize(ycrcb[1], ycrcb[1], y_res.size(), cv::INTER_CUBIC);
	cv::resize(ycrcb[2], ycrcb[2], y_res.size(), cv::INTER_CUBIC);

	cv::Mat SRres;
	cv::merge(ycrcb, SRres);
	cv::cvtColor(SRres, SRres, cv::COLOR_YCrCb2BGR);
	return SRres;
}

cv::Mat test_pic_2(const cv::Mat& img)
{
	auto net = std::make_shared<fsrcnnNet>(1, 4);

	torch::load(net, "../model/epoch_279.pt");

	//auto net = torch::jit::load("../model/epoch_279.pt");
	net->eval();

	if(torch::cuda::is_available())
		net->to(torch::kCUDA);
	std::vector<cv::Mat> ycrcb;

	cv::Mat Ycrcb;
	cv::cvtColor(img, Ycrcb, cv::COLOR_BGR2YCrCb);

	cv::split(Ycrcb, ycrcb);

	cv::Mat y = *ycrcb.begin();
	torch::Tensor tensor = torch::from_blob(y.data, {1, y.rows, y.cols, 1}, torch::kByte);
	tensor = tensor.permute({0,3,1,2});
	tensor = tensor.toType(torch::kFloat).div(255);

	if(torch::cuda::is_available())
		tensor = tensor.cuda();
	auto res = net->forward(tensor);

	cv::Mat y_res = Tensor_y_Tocv8uc1(res);

	ycrcb.front() = y_res.clone();
	cv::resize(ycrcb[1], ycrcb[1], y_res.size(), cv::INTER_CUBIC);
	cv::resize(ycrcb[2], ycrcb[2], y_res.size(), cv::INTER_CUBIC);
	

	cv::Mat SRres;
	cv::merge(ycrcb, SRres);

	cv::cvtColor(SRres, SRres, cv::COLOR_YCrCb2BGR);
	return SRres;
}

void train()
{
	//float lr;int nEpochs;int seed;int upscale_factor;int trainBatchSize;int testBatchSize;
	config params = {0.001, 300, 123, 4, 32, 16};
	FSRCNNTrainer trainer(params);

	std::string sourcePath = "../dataset/BSDS300/images/train/", labelPath = "../dataset/BSDS300/images/test/";
    std::vector<std::string> source, test;

    cv::glob(sourcePath, source);
    cv::glob(labelPath, test);

	fsrcnnDataset trainData(source,256,params.upscale_factor), testData(test,256,params.upscale_factor);

	trainer.run(trainData, testData);

	std::cout << "training over!" << std::endl;
}

int main(int argc, char** argv)
{
	cv::Mat img = cv::imread("../test_img/3096.jpg");
	cv::Mat res = test_pic_2(img);
	cv::imwrite("res.jpg", res);
	return 0;
}

