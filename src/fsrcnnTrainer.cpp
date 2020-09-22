#include "fsrcnnTrainer.h"
#include <torch/cuda.h>

FSRCNNTrainer::FSRCNNTrainer(config param)
{
	is_cuda_useful = torch::cuda::is_available();
	superParams = param;
}

FSRCNNTrainer::~FSRCNNTrainer()
{

}

void FSRCNNTrainer::build_modle()
{
	model = std::make_shared<fsrcnnNet>(1, 4);
	if(is_cuda_useful) model->to(torch::kCUDA);

	model->apply(init_weights);

	torch::manual_seed(superParams.seed);

	//if(is_cuda_useful)
		//torch::cuda::manual_seed(superParams.seed);
	 //torch::cuda::manual_seed(superParams.seed);
}

void FSRCNNTrainer::train(fsrcnnDataset trainData)
{
	auto criterion = torch::nn::MSELoss();
    auto myset = trainData.map(torch::data::transforms::Stack<>());
   	auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(superParams.lr));//0.0001


    auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(myset), superParams.trainBatchSize);
    //auto dataloader = torch::data::make_data_loader<>
    model->train();

    int batch_index = 0;
    float loss_sum(0.0);
    for(auto& batch : *dataloader)
    {
    	auto input = batch.data;
    	auto target = batch.target;
    	if(is_cuda_useful){
    		input = input.cuda();
    		target = target.cuda();
    	}
    	optimizer.zero_grad();
    	auto output = model->forward(input);
    	auto loss = criterion(output, target);
    	float lossv = loss.template item<float>();

    	loss.backward();
    	optimizer.step();
    	loss_sum += lossv;

    	if(batch_index++ % 50 == 0){
    		printf("batch: %d, loss: %.4f\n", batch_index, lossv);
    	}
    }
    printf("batch average loss: %.4f\n", loss_sum/trainData.size().value());
}

void FSRCNNTrainer::save_module(char* name)
{
	torch::save(model, name);
}

void FSRCNNTrainer::test(fsrcnnDataset testData)
{
	auto criterion = torch::nn::MSELoss();
    auto myset = testData.map(torch::data::transforms::Stack<>());
   
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(myset), superParams.testBatchSize);
    
    model->eval();

    float avg_psnr(0.0);

    torch::NoGradGuard no_grad;

    for(auto& batch : *dataloader)
    {
    	auto input = batch.data;
    	auto target = batch.target;
    	if(is_cuda_useful){
    		input = input.cuda();
    		target = target.cuda();
    	}
    	auto output = model->forward(input);
    	auto loss = criterion(output, target);
    	float mse = loss.template item<float>();
    	float psnr = 10 * std::log10(1 / mse);

    	avg_psnr += psnr;
    }
    printf("Average PSNR: %.4f dB\n", avg_psnr/testData.size().value());
}

void FSRCNNTrainer::run(fsrcnnDataset trainData, fsrcnnDataset testData)
{
	build_modle();
	for(int epoch = 0; epoch < superParams.nEpochs; epoch++)
	{
		printf("\n==> Epoch %d starts!\n", epoch);
		train(trainData);
		test(testData);
		if((epoch+1) % 40 == 0){
			char buf[128];
			//sprintf(buf, "../model/epoch_%d.pt", epoch+1);
			//save_module(buf);

			//char buf[128];
            sprintf(buf, "../model/epoch_%d.pt", epoch);

            torch::save(model,buf);
            printf("save epoch %d module over!\n", epoch+1);
		}
	}
}