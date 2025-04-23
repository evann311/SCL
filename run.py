import os
import copy
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


from scl.config import config_dict

from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule
import logging


from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy

import torch.distributed as dist

rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

log_format = f"%(asctime)s - RANK {rank} - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
log = logging.getLogger(__name__) # Sử dụng 'log' thay vì 'logger'

import argparse

class MyCluster(ClusterEnvironment):

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True

    def master_address(self):
        return os.environ['CHIEF_IP']

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def world_size(self):
        return int(os.environ['WORLD_SIZE'])

    def global_rank(self) -> int:
        return int(os.environ['RANK'])

    def local_rank(self) -> int:
        return int(os.environ['LOCAL_RANK'])

    def node_rank(self) -> int:
        return int(os.environ["INDEX"])

    def set_global_rank(self, rank: int) -> None:
        pass

    def set_world_size(self, size: int) -> None:
        pass

class MyDDPPlugin(DDPStrategy):

    def init_ddp_connection(self, global_rank = None, world_size = None) -> None:
        master_uri = "tcp://%s:%s" % (os.environ['CHIEF_IP'], os.environ['MASTER_PORT'])
        dist.init_process_group(
        backend=self.torch_distributed_backend,
        init_method=master_uri,
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='pretrain')
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    _config = copy.deepcopy(config_dict[config.task])
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = SCLTransformer(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    # model save setting
    if config.task == 'pretrain': 
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            verbose=True,
            monitor="val/the_metric",
            mode="max",
            save_last=True,
            every_n_train_steps=5000, # to save checkpoints each 5k steps according to val metrics
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/the_metric",
            mode="max",
            save_last=False,
        )

        last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=0,  
            verbose=True,
            save_last=True,  
        )


    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )
    log_dir = logger.log_dir

    profilter = PyTorchProfiler(
        dirpath=log_dir,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback, last_checkpoint_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    trainer = pl.Trainer(
        # plugins=[MyCluster(), MyDDPPlugin()], # for multi-machine ddp
        accelerator="gpu" if _config.get("num_gpus", 0) > 0 else "cpu",
        devices=_config.get("num_gpus", 1),
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        max_steps=150,
        callbacks=callbacks,
        logger=logger,
        profiler=profilter,
        accumulate_grad_batches=grad_steps,
        enable_model_summary=True,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        log_every_n_steps=1,
        # limit_train_batchs=5,
        # limit_val_batches=1
    )
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config.get("resume_from", None))
    else:
        trainer.test(model, datamodule=dm)
