#ifndef _TRANS_FORM_H_HH
#define _TRANS_FORM_H_HH

#include <opencv2/opencv.hpp>

inline int calculate_valid_crop_size(int crop_size, int upscale_factor)
{
	return crop_size - (crop_size % upscale_factor);
}

inline cv::Mat CenterCrop(const cv::Mat& img, int crop_size)
{	
	while(1){
		if(std::min(img.rows, img.cols) > crop_size) break;
		crop_size /= 2;
	}

	int x = (img.cols - crop_size) / 2;
	int y = (img.rows - crop_size) / 2;
	return img(cv::Rect(x,y,crop_size,crop_size)).clone();
}

inline cv::Mat Resize(const cv::Mat& img, int upscale_factor)
{	
	cv::Mat res;
	int min = std::min(img.cols, img.rows);
	if(min == img.cols){
		int output_h = img.rows / (float)img.cols * upscale_factor;
		cv::resize(img, res, cv::Size(upscale_factor, output_h));
	}
	else{
		int output_w = img.cols / (float)img.rows * upscale_factor;
		cv::resize(img, res, cv::Size(output_w, upscale_factor));
	}

	return res;
}

inline cv::Mat input_transform(const cv::Mat& img, int crop_size, int upscale_factor)
{
	return Resize(CenterCrop(img, crop_size), crop_size/upscale_factor);
}

inline cv::Mat target_transform(const cv::Mat& img, int crop_size)
{
	return CenterCrop(img, crop_size);
}

inline torch::Tensor cv8uc3ToTensor(cv::Mat frame)
{
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    frame.convertTo(frame, CV_32FC3/*, 1.0f / 255.0f*/);

    //auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3});
    auto input_tensor = torch::from_blob(frame.data, {frame.rows, frame.cols, 3});

    //input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor = input_tensor.permute({2, 0, 1});

    return input_tensor.div(255).toType(torch::kFloat);
}

inline cv::Mat TensorTocv8uc3(torch::Tensor out_tensor)
{

    out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);

    cv::Mat resultImg(out_tensor.size(0), out_tensor.size(1), CV_8UC3, out_tensor.data_ptr());
    cv::cvtColor(resultImg, resultImg, cv::COLOR_RGB2BGR);

    return resultImg;
}

inline torch::Tensor cv8uc3_y_to_Tensor(cv::Mat frame)
{
	cv::Mat yuv;
	cv::cvtColor(frame, yuv, cv::COLOR_BGR2YCrCb);
	
	std::vector<cv::Mat> ycrcb;	
	cv::split(yuv, ycrcb);

	cv::Mat y = *ycrcb.begin();
	torch::Tensor tensor = torch::from_blob(y.data, {y.rows, y.cols, 1}, torch::kByte);
	tensor = tensor.permute({2,0,1});
	return tensor.toType(torch::kFloat).div(255);
}

inline cv::Mat Tensor_y_Tocv8uc1(torch::Tensor x)
{
	std::cout << "tensor x size: " << x.sizes() << std::endl;
	auto tensor = x.squeeze(0).squeeze(0);
	
	int cols = tensor.size(1);
	int rows = tensor.size(0);

	std::cout << "rows: " << rows << "cols: " << cols << std::endl;

	tensor = tensor.mul(255).clamp(0,255).to(torch::kU8);
	tensor = tensor.to(torch::kCPU);
	cv::Mat res = cv::Mat(rows, cols, CV_8UC1, tensor.data_ptr());

	return res;
}
#endif