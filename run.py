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


    tb_handler_instance = torch.profiler.tensorboard_trace_handler(
        dir_name=os.path.join(_config["log_dir"], "profiler"),
        worker_name=None, #
    )
    


    def wrapped_on_trace_ready(profiler_instance):
        callback_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1))) # -1 nếu không có DDP
        log.debug(f"--- wrapped_on_trace_ready called on RANK {callback_rank} ---")

        # <<< !!! CHỈ RANK 0 MỚI THỰC HIỆN VIỆC GHI FILE !!! >>>
        if callback_rank == 0 or callback_rank == -1: # -1 là trường hợp không DDP
            try:
                # --- Xác định thư mục đích ---
                tb_dir = None
                if hasattr(profiler_instance, 'dir_name') and profiler_instance.dir_name:
                    # Ưu tiên dir_name nếu nó được cấu hình trực tiếp trong Profiler
                    # PyTorch Profiler thường lưu vào dir_name/worker_name
                    # TensorBoard Handler sẽ ghi vào thư mục được chỉ định cho nó
                    tb_dir = Path(profiler_instance.dir_name)
                    log.info(f"Rank {callback_rank}: Using dir_name from profiler instance: {str(tb_dir)}")
                elif hasattr(profiler_instance, 'profile') and hasattr(profiler_instance.profile, 'dir_name') and profiler_instance.profile.dir_name:
                    # Một số cấu hình có thể lồng profile object
                    tb_dir = Path(profiler_instance.profile.dir_name)
                    log.info(f"Rank {callback_rank}: Using dir_name from nested profiler.profile instance: {str(tb_dir)}")
                else:
                    # Nếu không lấy được từ profiler, sử dụng đường dẫn mặc định
                    log.warning(f"Rank {callback_rank}: Could not get dir_name from profiler instance. Falling back to default './result/plugins/profile'.")
                    tb_dir = Path("./result/plugins/profile") # <<< Đảm bảo đây là nơi bạn muốn

                log.info(f"Rank {callback_rank}: Target TensorBoard directory: {str(tb_dir)}")

                # --- Đảm bảo thư mục tồn tại và có quyền ghi ---
                log.debug(f"Rank {callback_rank}: Ensuring directory exists: {str(tb_dir)}")
                tb_dir.mkdir(parents=True, exist_ok=True)

                # Kiểm tra quyền ghi cơ bản (tùy chọn nhưng hữu ích)
                try:
                    test_file = tb_dir / f"write_test_rank_{callback_rank}.txt"
                    with open(test_file, "w") as f:
                        f.write("test")
                    test_file.unlink() # Xóa file test nếu thành công
                    log.debug(f"Rank {callback_rank}: Write permission test successful in {str(tb_dir)}")
                except Exception as write_err:
                    log.error(f"Rank {callback_rank}: Write permission test FAILED in {str(tb_dir)}", exc_info=True)
                    # Có thể dừng ở đây nếu không có quyền ghi
                    log.error(f"Rank {callback_rank}: Aborting trace processing due to write error.")
                    return # Không gọi handler nếu không ghi được

                # --- Tạo và gọi TensorBoard Handler ---
                log.info(f"Rank {callback_rank}: Creating TensorBoard handler for directory: {str(tb_dir)}")
                # worker_name giúp phân biệt các file trace nếu nhiều worker ghi vào cùng thư mục (không cần thiết nếu chỉ rank 0 ghi)
                # handler = torch.profiler.tensorboard_trace_handler(str(tb_dir), worker_name=f"rank_{callback_rank}")
                handler = torch.profiler.tensorboard_trace_handler(str(tb_dir))
                log.info(f"Rank {callback_rank}: Calling handler function...")
                handler(profiler_instance) # Gọi handler gốc
                log.info(f"Rank {callback_rank}: TensorBoard handler function finished successfully.")

                # --- Kiểm tra lại file kết quả ---
                log.debug(f"Rank {callback_rank}: Checking for event files in {str(tb_dir)} immediately after handler call...")
                # Handler của TensorBoard tạo file .tfevents
                tb_event_files = list(tb_dir.glob("*.tfevents.*"))
                if tb_event_files:
                    log.info(f"Rank {callback_rank}: Found TensorBoard event files: {[str(f) for f in tb_event_files]}")
                else:
                    log.warning(f"Rank {callback_rank}: NO TensorBoard event files (*.tfevents.*) found in {str(tb_dir)} immediately after handler call.")

                # Profiler cũng có thể tự tạo file .pt.trace.json hoặc .json (tùy phiên bản và cách cấu hình)
                # Việc này xảy ra *trước* khi on_trace_ready được gọi
                native_trace_files = list(tb_dir.glob("*.pt.trace.json")) + list(tb_dir.glob("*.json"))
                if native_trace_files:
                    log.info(f"Rank {callback_rank}: Found native profiler trace files (json/pt.trace.json): {[str(f) for f in native_trace_files]}")
                else:
                    log.warning(f"Rank {callback_rank}: NO native profiler trace files (*.json or *.pt.trace.json) found in {str(tb_dir)}.")


                # >>>>>>>>>> BẮT ĐẦU PHẦN ĐƯỢC COMMENT <<<<<<<<<<
                # Đoạn code dưới đây gây lỗi AttributeError vì profiler_instance (hoặc đối tượng profile bên trong nó)
                # không đảm bảo có thuộc tính 'filename' trong mọi trường hợp.
                # Việc kiểm tra file chrome trace cụ thể dựa trên tên file động này không đáng tin cậy.
                # Thay vào đó, chúng ta dựa vào việc tìm các file *.json hoặc *.pt.trace.json nói chung ở trên.
                #
                # chrome_trace_files = [
                # try:
                #     # Cố gắng truy cập filename một cách an toàn hơn, nhưng vẫn có thể thất bại
                #     profile_obj_with_filename = getattr(profiler_instance, 'profile', profiler_instance) # Thử cả profiler_instance.profile
                #     if hasattr(profile_obj_with_filename, 'filename') and profile_obj_with_filename.filename:
                #         expected_chrome_filename = f"{profile_obj_with_filename.filename}.json"
                #         log.debug(f"Rank {callback_rank}: Attempting to check for specific Chrome trace file: {expected_chrome_filename}")
                #         chrome_trace_files = list(tb_dir.glob(expected_chrome_filename))
                #         if chrome_trace_files:
                #             log.info(f"Rank {callback_rank}: Found expected Chrome trace file: {[str(f) for f in chrome_trace_files]}")
                #         else:
                #             log.warning(f"Rank {callback_rank}: Expected Chrome trace file ({expected_chrome_filename}) NOT found in {str(tb_dir)}.")
                #     else:
                #          log.warning(f"Rank {callback_rank}: Profiler instance (or its .profile attribute) does not have a 'filename' attribute. Skipping specific Chrome trace file check.")
                # except AttributeError:
                #      log.warning(f"Rank {callback_rank}: Encountered AttributeError while trying to access profiler filename. Skipping specific Chrome trace file check.")
                #
                # if not chrome_trace_files:
                #     # File .json này thường được tạo bởi chính Profiler *trước khi* gọi on_trace_ready.
                #     # Việc không tìm thấy nó có thể là một vấn đề khác hoặc không phải là vấn đề (nếu chỉ dùng tfevents).
                #     # Log này có thể hơi dư thừa sau khi đã kiểm tra *.json ở trên.
                #     log.warning(f"Rank {callback_rank}: Could not confirm specific Chrome trace file based on internal filename attribute.")
                # >>>>>>>>>> KẾT THÚC PHẦN ĐƯỢC COMMENT <<<<<<<<<<


            except Exception as e:
                log.error(f"Rank {callback_rank}: Error during TensorBoard handler execution in wrapped_on_trace_ready", exc_info=True)
        else:
            log.debug(f"Rank {callback_rank}: Skipping TensorBoard file writing on non-zero rank.")


    profilter = PyTorchProfiler(
        dirpath=os.path.join(_config["log_dir"], "profiler"),
        filename=f"{exp_name}_seed{_config['seed']}",
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=wrapped_on_trace_ready, # SỬ DỤNG HÀM BỌC CỦA BẠN

        profile_memory=True,
        with_stack=False,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
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
