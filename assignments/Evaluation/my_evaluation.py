import numpy as np
import pandas as pd
from collections import Counter


class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.acc = None
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True]) / len(correct)

        self.confusion_matrix = {}

        for label in self.classes_:
            tp = np.sum((self.actuals == label) & (self.predictions == label))
            fp = np.sum((self.actuals != label) & (self.predictions == label))
            tn = np.sum((self.actuals != label) & (self.predictions != label))
            fn = np.sum((self.actuals == label) & (self.predictions != label))
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        return

    def accuracy(self):
        if self.confusion_matrix is None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average="macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]

            try:
                prec = float(tp) / (tp + fp)
            except:
                prec = 0
        else:
            if average == 'micro':
                prec = self.accuracy()
            else:
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]

                    try:
                        prec_label = float(tp) / (tp + fp)
                    except:
                        prec_label = 0

                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(len(self.actuals))
                    else:
                        raise Exception("Unknown type of average.")
                    prec = prec_label * ratio

        return prec

    def recall(self, target=None, average="macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()

        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]

            try:
                rec = float(tp) / (tp + fn)
            except:
                rec = 0
        else:
            recs = []
            for label in self.classes_:
                tp = self.confusion_matrix[label]["TP"]
                fn = self.confusion_matrix[label]["FN"]

                try:
                    rec_label = float(tp) / (tp + fn)
                except:
                    rec_label = 0

                recs.append(rec_label)

            if average == 'macro':
                rec = np.mean(recs)
            elif average == "weighted":
                class_counts = [np.sum(self.actuals == label) for label in self.classes_]
                recalls_weighted = [rec * count for rec, count in zip(recs, class_counts)]
                rec = np.sum(recalls_weighted) / np.sum(class_counts)
            elif average == "micro":
                tp_sum = np.sum(
                    [np.sum((self.actuals == label) & (self.predictions == label)) for label in self.classes_])
                fn_sum = np.sum(
                    [np.sum((self.actuals == label) & (self.predictions != label)) for label in self.classes_])
                rec = tp_sum / (tp_sum + fn_sum)

        return rec

    def f1(self, target=None, average="macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        if target:
            prec = self.precision(target=target, average=average)
            rec = self.recall(target=target, average=average)

            try:
                f1_score = 2.0 * (prec * rec) / (prec + rec)
            except:
                f1_score = 0
        else:
            "write your own code"
            if average == 'micro':
                prec = self.precision(target=target, average=average)
                rec = self.recall(target=target, average=average)

                try:
                    f1_score = 2.0 * (prec * rec) / (prec + rec)
                except:
                    f1_score = 0
            elif average == 'macro' or average == 'weighted':
                f1_scores = []
                for label in self.classes_:
                    prec = self.precision(target=label, average=average)
                    rec = self.recall(target=label, average=average)

                    try:
                        f1_score_label = 2.0 * (prec * rec) / (prec + rec)
                    except:
                        f1_score_label = 0

                    f1_scores.append(f1_score_label)

                f1_score = np.mean(f1_scores)
            else:
                raise Exception("Unknown type of average.")

        return f1_score

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if self.pred_proba is None:
            return None
        else:
            y_true = np.where(self.actuals == target, 1.0, 0.0)
            axes = np.column_stack((self.pred_proba[target], y_true))
            sorted_axes = axes[np.argsort(axes[:, 0])[::-1]]

            tp = 0
            fp = 0
            prev_tpr = 0
            prev_fpr = 0
            auc_target = 0

            for target, y in sorted_axes:

                if y == 1:
                    tp += 1
                else:
                    fp += 1

                tpr = tp / sum(y_true)
                fpr = fp / (len(y_true) - sum(y_true))

                auc_target += (0.5 * (fpr - prev_fpr) * (tpr + prev_tpr))

                prev_tpr = tpr
                prev_fpr = fpr

            return auc_target

    def auc_old(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                prev_fpr = 0
                prev_tpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp += 1
                        tpr = tp / sum(self.pred_proba[target])
                    else:
                        fp += 1
                        fpr = fp / (len(order) - sum(self.pred_proba[target]))
                        auc_target += (0.5 * (fpr - prev_fpr) * (tpr + prev_tpr))

                        prev_fpr = fpr
                        prev_tpr = tpr
            else:
                raise Exception("Unknown target class.")

            return auc_target
