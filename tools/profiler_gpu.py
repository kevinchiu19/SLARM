import os
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler


class GPUTorchProfileHook():
    def __init__(self, mode, skip_first=3, wait=3, warmup=3, active=1, repeat=1, with_stack=False):
        my_schedule = schedule(skip_first=skip_first,
                               wait=wait,
                               warmup=warmup,
                               active=active,
                               repeat=repeat)

        profile_tb_logger = os.path.join("profile_results/{}".format(mode))
        os.makedirs(profile_tb_logger, exist_ok=True)

        self.profiler = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=my_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=with_stack,
                with_flops=False,
                with_modules=False,
                on_trace_ready=tensorboard_trace_handler(profile_tb_logger),
            )

    def before_train(self):
        self.profiler.start()

    def after_step(self):
        self.profiler.step()

    def after_train(self):
        self.profiler.stop()
