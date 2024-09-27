from .interfaces.registry import get_forward_proc_func

# from .interfaces.post_epoch_proc import default_post_epoch_process_without_teacher
# from .interfaces.post_forward_proc import default_post_forward_process
# from .interfaces.pre_epoch_proc import default_pre_epoch_process_without_teacher
# from .interfaces.pre_forward_proc import default_pre_forward_process
# from .interfaces.registry import get_pre_epoch_proc_func, get_pre_forward_proc_func, get_forward_proc_func, \
#     get_post_forward_proc_func, get_post_epoch_proc_func

from ..datasets.utils import build_data_loaders
from ..common.constant import def_logger
from ..losses.registry import get_high_level_loss, get_low_level_loss, get_mid_level_loss, get_func2extract_model_output
from ..optim.registry import get_optimizer, get_scheduler
# from tensorlayerx import nn
from torch import nn
from ..common.module_util import check_if_wrapped
from ..modules.utils import redesign_model
from ..common.module_util import get_module
from ..datasets.registry import get_dataset
from ..datasets.utils import extract_dataset_info
# from .interfaces.post_epoch_proc import default_pre_epoch_process_without_teacher
from tensorlayerx.model import TrainOneStep

logger = def_logger.getChild(__name__)


class TrainingBox(object):
    def setup_data_flows(self, train_config):
        data_loader_dict = dict()
        use_dataloader = train_config['use_dataloader']
        if use_dataloader:
            train_data_loader_config = train_config.get('train_data_loader', dict())
            val_data_loader_config = train_config.get('val_data_loader', dict())
            test_data_loader_config = train_config.get('test_data_loader', dict())
        
            data_loader_dict = build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config, test_data_loader_config])

            if data_loader_dict['train'] is not None:
                self.train_data_loader = data_loader_dict['train']
            if data_loader_dict['val'] is not None:
                self.val_data_loader = data_loader_dict['val']
            if data_loader_dict['test'] is not None:
                self.val_data_loader = data_loader_dict['test']
        else:
            data_dict = build_data_loaders(self.dataset_dict, use_dataloader=False)
            if data_dict['train'] is not None:
                self.train_data = data_dict['train']
            if data_dict['val'] is not None:
                self.val_data = data_dict['val']
            if data_dict['test'] is not None:
                self.test_data = data_dict['test']
    
    def setup_data(self):
        data = dict()
        dataset = self.dataset_dict
        dataset = get_dataset(dataset['key'], **dataset['init']['kwargs'])
        graph = dataset[0]
        data.update(extract_dataset_info(dataset))
        data.update(extract_dataset_info(graph))
        
        self.data = data

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        if criterion_config.get('kwargs', None) is None:
            criterion_config['kwargs'] = dict()
        criterion_config['kwargs']['net'] = self.model
        # criterion_config['net'] = self.model
        self.criterion = get_high_level_loss(criterion_config)
        logger.info(self.criterion)
        # self.extract_model_loss = get_func2extract_model_output(criterion_config.get('func2extract_model_loss', None))

    def setup_model(self, model_config):
        # TODO: 设计hook机制，从checkpoint加载模型
        unwrapped_org_model = self.org_model.module if check_if_wrapped(self.org_model) else self.org_model
        ref_model = unwrapped_org_model
        if len(model_config) > 0 or (len(model_config) == 0 and self.model is None):
            model_type = 'original'
            self.model = redesign_model(ref_model, model_config, 'student', model_type)

        self.model_forward_proc = get_forward_proc_func(model_config.get('forward_proc', None))
    
    def setup_train_one_step(self):
        loss_func = self.criterion
        optimizer = self.optimizer
        train_weights = self.model.trainable_weights
        self.train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    # def setup_pre_post_processes(self, train_config):
    #     """
    #     Sets up pre/post-epoch/forward processes for the current training stage.
    #     This method will be internally called when instantiating this class and when calling
    #     :meth:`MultiStagesTrainingBox.advance_to_next_stage`.

    #     :param train_config: training configuration.
    #     :type train_config: dict
    #     """
    #     pre_epoch_process = default_pre_epoch_process_without_teacher
    #     if 'pre_epoch_process' in train_config:
    #         pre_epoch_process = get_pre_epoch_proc_func(train_config['pre_epoch_process'])
    #     setattr(TrainingBox, 'pre_epoch_process', pre_epoch_process)
    #     pre_forward_process = default_pre_forward_process
    #     if 'pre_forward_process' in train_config:
    #         pre_forward_process = get_pre_forward_proc_func(train_config['pre_forward_process'])
    #     setattr(TrainingBox, 'pre_forward_process', pre_forward_process)
    #     post_forward_process = default_post_forward_process
    #     if 'post_forward_process' in train_config:
    #         post_forward_process = get_post_forward_proc_func(train_config['post_forward_process'])

    #     setattr(TrainingBox, 'post_forward_process', post_forward_process)
    #     post_epoch_process = default_post_epoch_process_without_teacher
    #     if 'post_epoch_process' in train_config:
    #         post_epoch_process = get_post_epoch_proc_func(train_config['post_epoch_process'])
    #     setattr(TrainingBox, 'post_epoch_process', post_epoch_process)

    def setup(self, train_config):
        self.setup_data_flows(train_config)

        self.setup_data()

        model_config = train_config.get('model', dict())
        self.setup_model(model_config)

        self.setup_loss(train_config)

        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_kwargs = optim_config['kwargs']

            module_wise_configs = optim_config.get('module_wise_kwargs', list())
            if len(module_wise_configs) > 0:
                trainable_module_list = list()
                for module_wise_config in module_wise_configs:
                    module_wise_kwargs = dict()
                    if isinstance(module_wise_config.get('kwargs', None), dict):
                        module_wise_kwargs.update(module_wise_config['kwargs'])

                    module = get_module(self.model, module_wise_config['module'])
                    module_wise_kwargs['params'] = module.parameters() if isinstance(module, nn.Module) else [module]
                    trainable_module_list.append(module_wise_kwargs)
            else:
                trainable_module_list = self.model.trainable_weights
                
            filters_params = optim_config.get('filters_params', True)
            self.optimizer = get_optimizer(trainable_module_list, optim_config['key'], **optim_kwargs, filters_params=filters_params)
            # self.optimizer.zero_grad()

        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['key'], **scheduler_config['kwargs'])
            self.scheduling_step = scheduler_config.get('scheduling_step', 0)
        elif optimizer_reset:
            self.lr_scheduler = None
            self.scheduling_step = None

        # self.setup_pre_post_processes(train_config)
        self.setup_train_one_step()


    def __init__(self, model, dataset_dict, train_config):
        # Key attributes (should not be modified)
        self.org_model = model
        self.dataset_dict = dataset_dict
        self.data = None
        # Local attributes (can be updated at each stage)
        self.model = None
        self.model_forward_proc = None
        self.target_model_pairs = list()
        self.model_io_dict = dict()
        self.train_data_loader, self.val_data_loader, self.test_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None, None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.criterion, self.extract_model_loss = None, None
        self.model_any_frozen = None
        self.grad_accum_step = None
        self.max_grad_norm = None
        self.scheduling_step = 0
        self.stage_grad_count = 0
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def forward_process(self, data, targets=None, **kwargs):
        # model_outputs = self.model_forward_proc(self.model, data, **kwargs)
        loss = self.criterion(data, targets)
        return loss
    
    def train(self, data, targets, **kwargs):
        self.model.set_train()
        train_loss = self.train_one_step(data, targets)

        return train_loss

    def pre_epoch_process(self, *args, **kwargs):
        raise NotImplementedError()


def get_training_box(model, dataset_dict, train_config):
    # if 'stage1' in train_config:
    #     return MultiStagesTrainingBox(model, dataset_dict,
    #                                   train_config, device, device_ids, distributed, lr_factor, accelerator)
    return TrainingBox(model, dataset_dict, train_config)
