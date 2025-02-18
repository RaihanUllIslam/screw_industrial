import os
import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epochs, learning_rate, grad_clip_value, device, weight_decay=1e-4, early_stopping_patience=25):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        self.device = device
        self.best_loss = float('inf')
        self.train_losses = [] 
        self.val_losses = []
        self.weight_decay = weight_decay 
        self.early_stopping_patience = early_stopping_patience  
        self.patience_counter = 0 

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()

                if self.grad_clip_value:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            val_loss = self.validate(criterion)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0 
                print(f"New best model saved with validation loss {self.best_loss:.6f}")
                model_save_dir = './trained_model'
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                torch.save(self.model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))

            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered. No improvement in validation loss for {self.early_stopping_patience} epochs.")
                    break
        self.plot_losses()

    def validate(self, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        return val_loss

    def plot_losses(self):
        eval_results_dir = './eval_results'
        if not os.path.exists(eval_results_dir):
            os.makedirs(eval_results_dir)
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(eval_results_dir, 'loss_curve.png'))
        plt.close()

        print(f"Loss curves saved in {eval_results_dir}/loss_curve.png")
