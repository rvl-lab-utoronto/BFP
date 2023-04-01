class Callback:
    """Base class for callback."""

    def __init__(self, args, model, dataset):
        pass

    def on_train_batch_end(self, model, batch):
        pass

    def on_train_epoch_end(self, epoch, model, task_train_loader):
        pass

    def on_task_end(self, task_id, model, task_train_loader):
        pass

    def on_train_end(self, model, train_loader_all, test_loader_all):

        pass