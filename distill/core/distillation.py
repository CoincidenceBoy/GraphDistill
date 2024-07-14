from distill.common.constant import def_logger

from distill.common.main_util import load_ckpt
from distill.core.interfaces.registry import get_forward_proc_func
from distill.core.util import set_hooks
from distill.modules.utils import redesign_model
from distill.losses.registry import get_high_level_loss

logger = def_logger.getChild(__name__)

class DistillationBox(object):
    def __init__(self, teacher_model, student_model, dataset_dict, train_config, device):
        # Key attributes (should not be modified)
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.dataset_dict = dataset_dict
        self.device = device
        # Local attributes (can be updated at each stage)
        self.teacher_model = None
        self.student_model = None
        self.teacher_forward_proc, self.student_forward_proc = None, None
        self.target_teacher_pairs, self.target_student_pairs = list(), list()
        self.teacher_io_dict, self.student_io_dict = dict(), dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.criterion, self.extract_model_loss = None, None
        self.teacher_updatable, self.teacher_any_frozen, self.student_any_frozen = None, None, None

        self.num_epochs = train_config['num_epochs']
        self.setup(train_config)

    def setup(self, train_config):
        # Define teacher and student models used in this stage
        teacher_config = train_config.get('teacher', dict())
        student_config = train_config.get('student', dict())
        self.setup_teacher_student_models(teacher_config, student_config)

        # Define loss function used in this stage
        self.setup_loss(train_config)

    def setup_teacher_student_models(self, teacher_config, student_config):
        teacher_ref_model = self.org_teacher_model
        student_ref_model = self.org_student_model
        if len(teacher_config) > 0 or (len(teacher_config) == 0 and self.teacher_model is None):
            logger.info('[teacher model]')
            model_type = 'original'
            self.teacher_model = redesign_model(teacher_ref_model, teacher_config, 'teacher', model_type)
            src_teacher_ckpt_file_path = teacher_config.get('src_ckpt', None)
            if src_teacher_ckpt_file_path is not None:
                load_ckpt(src_teacher_ckpt_file_path, self.teacher_model)

        if len(student_config) > 0 or (len(student_config) == 0 and self.student_model is None):
            logger.info('[student model]')
            model_type = 'original'
            self.student_model = redesign_model(student_ref_model, student_config, 'student', model_type)
            src_student_ckpt_file_path = student_config.get('src_ckpt', None)
            if src_student_ckpt_file_path is not None:
                load_ckpt(src_student_ckpt_file_path, self.student_model)

        self.teacher_any_frozen = \
            len(teacher_config.get('frozen_modules', list())) > 0 or not teacher_config.get('requires_grad', True)
        self.student_any_frozen = \
            len(student_config.get('frozen_modules', list())) > 0 or not student_config.get('requires_grad', True)

        self.target_teacher_pairs.extend(set_hooks(self.teacher_model, teacher_ref_model,
                                                   teacher_config, self.teacher_io_dict))
        self.target_student_pairs.extend(set_hooks(self.student_model, student_ref_model,
                                                   student_config, self.student_io_dict))
        self.teacher_forward_proc = get_forward_proc_func(teacher_config.get('forward_proc', None))
        self.student_forward_proc = get_forward_proc_func(student_config.get('forward_proc', None))




    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        self.criterion = get_high_level_loss(criterion_config)
        logger.info(self.criterion)
        # todo
        # self.extract_model_loss = get_func2extract_model_output(criterion_config.get('func2extract_model_loss', None))
