from torch import device
import torch

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
        
    if class_label >= num_classes:
        raise ValueError("The class label cannot be larger then the number of classes")
    
    tp = 0; tn = 0; fp = 0; fn = 0;
    tp = confusion_matrix[class_label][class_label]
    fp = confusion_matrix[:, class_label].sum() - tp
    fn = confusion_matrix[class_label, :].sum() - tp
    tn = confusion_matrix.sum() - tp - fp - fn 
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # guarding in case denominator is 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def f1_score_per_class(confusion_matrix, class_to_idx):
    f1_score_per_class = []
    for class_name in class_to_idx.keys():
        class_label = class_to_idx[class_name]
        precision, recall, f1_score = compute_macros_for_class(confusion_matrix, class_label)
        f1_score_per_class.append((class_name, f1_score))
        
    return f1_score_per_class
    
