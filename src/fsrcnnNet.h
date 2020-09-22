#ifndef _FSRCNN_NET_H_HH
#define _FSRCNN_NET_H_HH

#include <torch/torch.h>

inline void init_weights(torch::nn::Module& module)
{
	//stop autograd's work
	torch::NoGradGuard no_grad;
	if(auto* conv = module.as<torch::nn::Conv2d>()){
		torch::nn::init::normal_(conv->weight, 0.0, 0.2);
		if(conv->options.bias())
			torch::nn::init::zeros_(conv->bias);
	}
	else if(auto* convTrans = module.as<torch::nn::ConvTranspose2d>()){
		torch::nn::init::normal_(convTrans->weight, 0.0, 0.0001);
		if(convTrans->options.bias())
			torch::nn::init::zeros_(convTrans->bias);
	}
}

class simpleNet : public torch::nn::Module{
public:
	simpleNet()
	{
		auto c1 = torch::nn::Conv2dOptions(3, 12, 3).padding(2).stride(1).bias(false);

		conv = register_module("conv1", torch::nn::Conv2d(c1));

		auto c2 = torch::nn::Conv2dOptions(2, 2, 1).padding(0).stride(1);
		conv1 = register_module("conv2", torch::nn::Conv2d(c2));

		linear = register_module("linear1", torch::nn::Linear(256, 10));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return conv->forward(x);
	}
private:
	torch::nn::Conv2d conv{nullptr}, conv1{nullptr};
	torch::nn::Linear linear{nullptr};
};

class fsrcnnNet : public torch::nn::Module{
public:
	fsrcnnNet(int num_channels, int update_factor, int d = 64, int s = 12)
	{
		auto c1 = torch::nn::Conv2dOptions(num_channels, d, 5).padding(2).stride(1);
		//auto c1 = torch::nn::Conv2dOptions(3, 12, 3).padding(2).stride(1);

		conv1 = register_module("conv1", torch::nn::Conv2d(c1));
		#if 1
		auto c2 = torch::nn::Conv2dOptions(d, s, 1).padding(0).stride(1);
		conv2 = register_module("conv2", torch::nn::Conv2d(c2));

		auto c3 = torch::nn::Conv2dOptions(s, s, 3).padding(1).stride(1);
		conv3 = register_module("conv3", torch::nn::Conv2d(c3));
		conv4 = register_module("conv4", torch::nn::Conv2d(c3));
		conv5 = register_module("conv5", torch::nn::Conv2d(c3));
		conv6 = register_module("conv6", torch::nn::Conv2d(c3));

		auto c4 = torch::nn::Conv2dOptions(s, d, 1).padding(0).stride(1);
		conv7 = register_module("conv7", torch::nn::Conv2d(c4));

		auto c5 = torch::nn::ConvTranspose2dOptions(d, num_channels, 9).stride(update_factor).padding(3).output_padding(1);
		convTrans = register_module("convTrans", torch::nn::ConvTranspose2d(c5));
		#endif
		auto p_relu = torch::nn::PReLUOptions().num_parameters((1));
		prelu = register_module("prelu", torch::nn::PReLU(p_relu));
	}

	virtual ~fsrcnnNet(){}
public:
	torch::Tensor forward(torch::Tensor input)
	{	
		auto res = prelu->forward(conv1->forward(input));
		res = prelu->forward(conv2->forward(res));
		res = conv4->forward(conv3->forward(res));
		res = conv6->forward(conv5->forward(res));
		res = prelu->forward(res);
		res = prelu->forward(conv7->forward(res));
		return convTrans(res);
	}
private:
	//torch::nn::Sequential sequential{nullptr};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr},
					  conv5{nullptr}, conv6{nullptr}, conv7{nullptr};

	torch::nn::ConvTranspose2d convTrans{nullptr};

	torch::nn::PReLU prelu{nullptr};
};

#endif