import numpy as np

from sklearn.metrics import  recall_score, precision_score, confusion_matrix



class Threshold:
    def __init__(self, predict_proba_val, y):
        self.predict_proba_val = predict_proba_val
        self.y = y
      
    def find_threshold_with_constraints(self, max_false_negatives: int, min_accuracy: float = 0.7):
        """Search thresholds to satisfy BOTH constraints on validation set:
        - accuracy >= min_accuracy
        - false negatives (FN) <= max_false_negatives

        Returns (threshold, metrics_dict) where metrics_dict includes: accuracy, recall, precision, fn, fp, tp, tn.
        If multiple thresholds satisfy constraints, picks the one with lowest false negatives, then highest accuracy.
        If none satisfy constraints, returns None, {}.
        """
       
        probs = self.predict_proba_val()
        y_true = self.y
        thresholds = np.linspace(0,1,501)
        candidates = []
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            if acc < min_accuracy or fn > max_false_negatives:
                continue
            rec = recall_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            candidates.append({
                'threshold': thr,
                'accuracy': acc,
                'recall': rec,
                'precision': prec,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            })
        if not candidates:
            return None, {}
        # Sort: lowest fn, then highest accuracy, then highest recall
        candidates.sort(key=lambda d: (d['fn'], -d['accuracy'], -d['recall']))
        best = candidates[0]
        return best['threshold'], best


