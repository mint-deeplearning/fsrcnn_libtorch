#ifndef _FSRCNN_TRAINER_H_HH
#define _FSRCNN_TRAINER_H_HH

#include "fsrcnnNet.h"
#include "datasets.h"

typedef struct _configs{
	float lr;
	int nEpochs;
	int seed;
	int upscale_factor;
	int trainBatchSize;
	int testBatchSize;
}config;

class FSRCNNTrainer{
public:
	FSRCNNTrainer(config param);
	virtual ~FSRCNNTrainer();
public:
	void run(fsrcnnDataset trainData, fsrcnnDataset testData);
protected:
	void build_modle();
	void save_module(char* name);
	void train(fsrcnnDataset trainData);
	void test(fsrcnnDataset testData);
private:
	bool is_cuda_useful;
	config superParams;
	std::shared_ptr<fsrcnnNet> model;
};
#endif