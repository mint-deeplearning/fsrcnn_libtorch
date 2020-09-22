#ifndef _DATASETS_H_HH
#define _DATASETS_H_HH

#include <torch/torch.h>
#include "transform.h"

class fsrcnnDataset : public torch::data::datasets::Dataset<fsrcnnDataset> {
public:
	fsrcnnDataset(std::vector<std::string> source, int crop_size, int upscale_factor)
	{
		_Datasource = source;
		_crop_size = calculate_valid_crop_size(crop_size, upscale_factor);
		_upscale_factor = upscale_factor;
	}
	virtual ~fsrcnnDataset(){}
public:
	torch::data::Example<> get(size_t index)override
    {   
        cv::Mat s0 = cv::imread(_Datasource[index]);
        cv::Mat target = s0.clone();

        s0 = input_transform(s0, _crop_size, _upscale_factor);
        //auto tensor0 = cv8uc3ToTensor(s0);
        auto tensor0 = cv8uc3_y_to_Tensor(s0);


        target = target_transform(target, _crop_size);
        auto tensor1 = cv8uc3_y_to_Tensor(target);

  		//auto tensor1 = cv8uc3ToTensor(target);
        return {tensor0, tensor1};
    }

    torch::optional<size_t> size()const override {
        return _Datasource.size();
    };
private:
	std::vector<std::string> _Datasource;
	int _crop_size;
	int _upscale_factor;
};
#endif