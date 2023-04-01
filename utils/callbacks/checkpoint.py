import os
import torch

from .base import Callback


class CheckpointCallback(Callback):
    '''Save model checkpoint after each task'''
    def __init__(self, args, model, dataset):
        super().__init__(args, model, dataset)
        self.args = args
        self.save_folder = os.path.join(args.ckpt_folder, "weights", args.exp_name)
        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def on_task_end(self, task_id, model, task_train_loader):
        save_path = os.path.join(self.save_folder, "model_%d.ckpt" % task_id)
        print("Saving model weights to", save_path)
        to_save = {
            "state_dict": model.state_dict(),
            "args": self.args,
        }

        # if hasattr(model, "buffer"):
        #     buffer_data = list(model.buffer.get_all_data())
        #     for k, v in enumerate(buffer_data):
        #         buffer_data[k] = v.cpu()
        #     to_save["buffer"] = buffer_data

        torch.save(to_save, save_path)