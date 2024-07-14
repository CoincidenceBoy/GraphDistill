from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

class DistillationBox(object):
    def __init__(self, teacher_model, student_model, dataset_dict, train_config, device):
        # Key attributes (should not be modified)
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.dataset_dict = dataset_dict
        self.device = device
        # Local attributes (can be updated at each stage)
        # Local attributes (can be updated at each stage)
        self.teacher_model = None
        self.student_model = None
        self.teacher_forward_proc, self.student_forward_proc = None, None
        self.target_teacher_pairs, self.target_student_pairs = list(), list()
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



    def setup_loss(self, train_config):
        pass
