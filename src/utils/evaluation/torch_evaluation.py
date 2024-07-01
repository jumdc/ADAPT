"""Evaluation of the predictions."""

import torch
import torchmetrics
from torch import nn


class Evaluation(nn.Module):
    """Evaluation module."""

    def __init__(self, num_classes=2, logger=None):
        """init."""
        super().__init__()
        self.logger = logger
        task = "binary"
        average = None if num_classes == 2 else "macro"
        metric_params = {
            "task": task,
            "num_classes": num_classes,
            "average": average,
            "top_k": 1,
        }
        self.accuracy = torchmetrics.Accuracy(**metric_params)
        self.roc = torchmetrics.ROC(task="binary", num_classes=2)
        self.auc = torchmetrics.AUROC(task="binary", num_classes=2)
        self.f1 = torchmetrics.F1Score(**metric_params)
        self.f1_weighted = torchmetrics.F1Score(
            **{
                "task": "multiclass",
                "num_classes": num_classes,
                "top_k": 1,
                "average": "weighted",
            }
        )
        self.recall = torchmetrics.Recall(**{"task": "binary"})
        self.specifity = torchmetrics.Specificity(**{"task": "binary"})
        self.balanced_accuracy = torchmetrics.Accuracy(
            **{
                "task": "multiclass",
                "num_classes": num_classes,
                "top_k": 1,
                "average": "macro",
            }
        )

    def forward(self, y_pred, y_true, prefix):
        """forward method.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y_true : torch.Tensor
            True labels.
        prefix : str
            Prefix for the metrics.

        Returns
        -------
        metrics : dict
            Dictionary with the metrics.
        """
        metrics = {}
        auc_score = self.auc(y_pred, y_true)
        metrics[f"{prefix}auc"] = auc_score
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        metrics["acc"] = self.accuracy(y_pred, y_true)
        metrics[f"{prefix}test_f1"] = self.f1(y_pred, y_true)
        metrics[f"{prefix}recall"] = self.recall(y_pred, y_true)
        metrics[f"{prefix}specifity"] = self.specifity(y_pred, y_true)
        metrics[f"{prefix}balanced_accuracy"] = self.balanced_accuracy(y_pred, y_true)
        metrics[f"{prefix}f1_weighted"] = self.f1_weighted(y_pred, y_true)
        return metrics
