from sklearn.metrics import f1_score, classification_report, confusion_matrix


LABELS = [0, 1, 2, 3]
TARGET_NAMES = ["N", "A", "O", "~"]  # 0:N, 1:A, 2:O, 3:~


def evaluate(y_true, y_pred, *, verbose: bool = True):
    """
    Evaluate 4-class ECG classification (N, A, O, ~).

    Returns a dict with:
      - macro_f1
      - per_class_f1 (dict)
      - confusion_matrix (list of lists)
      - report (string)
    """
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    per_class_f1_values = f1_score(y_true, y_pred, labels=LABELS, average=None)

    per_class_f1 = {name: float(score) for name, score in zip(TARGET_NAMES, per_class_f1_values)}
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )

    if verbose:
        print(f"Macro F1: {macro_f1:.4f}")
        print("Per-class F1:", {k: f"{v:.4f}" for k, v in per_class_f1.items()})
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix (rows=true, cols=pred) in order [N, A, O, ~]:")
        print(cm)

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
