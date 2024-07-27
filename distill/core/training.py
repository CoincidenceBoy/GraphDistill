

class TrainingBox(object):
    def __init__(self, model, dataset_dict, train_config, device, lr_factor, accelerator=None):
        self.model_forward_proc = None
        self.setop(train_config)
        super.__init__()

    def setup_data_loaders(self, train_config):
        pass

    def setup(self, train_config):
        self.setup_data_loaders(train_config)

        model_config = train_config.get('model', dict())

    def foward_process(self, sample_batch, targets=None, supp_dict=None, **kwargs):
        model_outputs = self.model_forward_proc(self.model, sample_batch, targets, supp_dict, **kwargs)

    def pre_epoch_process(self, *args, **kwargs):
        raise NotImplementedError()




def get_training_box(model, dataset_dict, train_config, device, device_ids, distributed,
                     lr_factor, accelerator=None):
    # if 'stage1' in train_config:
    #     return MultiStagesTrainingBox(model, dataset_dict,
    #                                   train_config, device, device_ids, distributed, lr_factor, accelerator)
    return TrainingBox(model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator)
