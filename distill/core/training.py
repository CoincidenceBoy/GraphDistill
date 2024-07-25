

class TrainingBox(object):
    def __init__(self):
        super.__init__()




def get_training_box(model, dataset_dict, train_config, device, device_ids, distributed,
                     lr_factor, accelerator=None):
    # if 'stage1' in train_config:
    #     return MultiStagesTrainingBox(model, dataset_dict,
    #                                   train_config, device, device_ids, distributed, lr_factor, accelerator)
    return TrainingBox(model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator)
