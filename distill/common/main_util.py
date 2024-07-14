from .constant import def_logger

logger = def_logger.getChild(__name__)
def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    pass