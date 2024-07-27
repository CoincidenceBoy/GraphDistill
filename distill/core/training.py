from .interfaces.registry import get_forward_proc_func
from ..datasets.utils import build_data_loaders
from ..common.constant import def_logger
from ..losses.registry import get_high_level_loss, get_func2extract_model_output

logger = def_logger.getChild(__name__)

class TrainingBox(object):
    def setup_data_loaders(self, train_config):
        train_data_loader_config = train_config.get('train_data_loader', dict())
        # if 'requires_supp' not in train_data_loader_config:
        #     train_data_loader_config['requires_supp'] = True

        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader =\
            build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config])
        if train_data_loader is not None:
            self.train_data_loader = train_data_loader
        if val_data_loader is not None:
            self.val_data_loader = val_data_loader

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        self.criterion = get_high_level_loss(criterion_config)
        logger.info(self.criterion)
        self.extract_model_loss = get_func2extract_model_output(criterion_config.get('func2extract_model_loss', None))

    def setup_model(self, model_config):
        # TODO: 设计hook机制，从checkpoint加载模型
        self.model_forward_proc = get_forward_proc_func(model_config.get('forward_proc', None))

    def setup(self, train_config):
        self.setup_data_loaders(train_config)

        model_config = train_config.get('model', dict())
        self.setup_model(model_config)

        self.setup_loss(train_config)


    def __init__(self, model, dataset_dict, train_config, device, lr_factor, accelerator=None):
        self.model_forward_proc = None
        self.setop(train_config)
        super.__init__()

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
