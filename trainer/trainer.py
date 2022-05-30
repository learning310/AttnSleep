import torch
import numpy as np
from base import BaseTrainer
from utils import MetricTracker

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # 模型训练
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, self.class_weights, self.device)
            loss.backward()
            self.optimizer.step()

            # 更新指标
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})  # NOTE: 在各个项目前加入 'val_' 前缀以作区分

            if val_log["accuracy"] > self.selected: # 记录之前在所有迭代中模型在验证集上最优表现结果，并记录结果
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs

            if epoch == total_epochs: # 迭代结束的时候返回记录的最好的结果
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
        return self.valid_metrics.result(), outs, trgs
