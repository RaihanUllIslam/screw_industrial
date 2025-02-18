import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

class Evaluator:
    def __init__(self, model, test_loader, output_dir='./eval_results', device='cpu'):
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def evaluate(self):  
        all_labels = []
        all_preds = []

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        self.save_metrics(accuracy, precision, recall, f1)

        class_names = ['good', 'not-good']
        self.plot_confusion_matrix(all_labels, all_preds, class_names)

    def save_metrics(self, accuracy, precision, recall, f1):
        with open(os.path.join(self.output_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print(f"Evaluation results saved to {self.output_dir}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                         annot_kws={"color": "black"}, xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

        print(f"Confusion matrix saved to {self.output_dir}")
