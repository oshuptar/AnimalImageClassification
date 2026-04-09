from torch import device
import torch
import os
import csv

def get_device() -> device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device

# rows = true labels. columns = predicted labels
def compute_macros_for_class(confusion_matrix, class_label: int) -> tuple[float, float, float]:
    num_classes = len(confusion_matrix)
    for row in confusion_matrix:
        if len(row) != num_classes:
            raise ValueError("The dimension of confusion matrix do not match")
        
    if class_label >= num_classes or class_label < 0:
        raise ValueError("The class label cannot be larger then the number of classes")
    
    tp = 0; tn = 0; fp = 0; fn = 0;
    tp = confusion_matrix[class_label, class_label]
    fp = confusion_matrix[:, class_label].sum() - tp
    fn = confusion_matrix[class_label, :].sum() - tp
    tn = confusion_matrix.sum() - tp - fp - fn 
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # guarding in case denominator is 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def compute_f1_score_per_class(confusion_matrix, class_to_idx):
    f1_score_per_class = []
    for class_name in class_to_idx.keys():
        class_label = class_to_idx[class_name]
        precision, recall, f1_score = compute_macros_for_class(confusion_matrix, class_label)
        f1_score_per_class.append((class_name, f1_score))
        
    return f1_score_per_class

def compute_f1_macro(f1_scores):
    f1_macro = 0.0
    num_classes = len(f1_scores)
    for _, f1_score in f1_scores:
        f1_macro += f1_score
    f1_macro = f1_macro/num_classes
    return f1_macro

def row_normalise_confusion_matrix(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(float).copy()
    num_classes = len(confusion_matrix)
    for i in range(num_classes):
        row_sum = confusion_matrix[i, :].sum()
        if row_sum > 0:
            confusion_matrix[i, :] = confusion_matrix[i, :] / row_sum
    return confusion_matrix

def save_test_results(folder_name: str, filename: str, history: list[dict], class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(root_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, filename + '.csv')
    
    with open(output_path, mode="w", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["filename", "actual", "top_1", "top_5"])

        for record in history:
            probabilities = record["probabilities"]
            class_probability_pairs = [
                (idx_to_class[i], float(probabilities[i]))
                for i in range(len(probabilities))
            ]
            sorted_probabilities = sorted(
                class_probability_pairs,
                key=lambda x: x[1],
                reverse=True
            )
            top1 = sorted_probabilities[0]
            top5 = sorted_probabilities[:5]
            top1_class, top1_prob = top1
            top5_str = " | ".join(
                f"{class_name}:{class_prob}" for class_name, class_prob in top5
            )
            csv_writer.writerow([
                record["filename"],
                idx_to_class[record["actual"]],
                f"{top1_class}:{top1_prob:.4f}",
                top5_str
            ])

    print(f"Results were successfully written to {output_path}")


def get_filter_size():
    return [32, 64]

def get_learning_rate():
    return [1e-3, 3e-4]

def get_batch_size():
    return [64, 128]

def get_weight_decay():
    return [0, 1e-4]
    
