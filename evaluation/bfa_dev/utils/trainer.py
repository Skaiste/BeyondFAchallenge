import torch
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, valid_loader, task, optimizer,
                num_epochs, device, output_dir='results'):
    
    writer = SummaryWriter(log_dir=f'{output_dir}/runs')
    model.to(device)

    if task == 'age':
        criterion = nn.L1Loss()
    elif task == 'sex':
        criterion = nn.BCELoss()
    elif task == 'cognitive_status':
        criterion = nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f'Invalid task value: {task}.')

    # Track best model
    best_valid_metric = float('inf') if task == 'age' else 0.0  # Lower is better for MAE, higher for accuracy
    best_epoch = 0
    best_train_loss = 0.0
    best_train_metric = 0.0
    best_valid_loss = 0.0
    best_valid_metric_value = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0.0
        predictions, ground_truths = [], []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            if task != 'cognitive_status':
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            if task == 'cognitive_status':
                outputs_extend = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            elif task == 'sex':
                outputs_extend = (outputs.detach().cpu().numpy() > 0.5).astype(int)
            else:
                outputs_extend = outputs.detach().cpu().numpy()
            predictions.extend(outputs_extend)
            ground_truths.extend(targets.cpu().numpy())

        avg_loss_train = total_loss_train / len(train_loader)
        if task in ['sex', 'cognitive_status']:
            eval_train = accuracy_score(ground_truths, predictions)
        else:
            eval_train = mean_absolute_error(ground_truths, predictions)
            # eval_train = (abs(predictions - ground_truths) / ground_truths).mean()

        print(f"==TRAIN== Epoch {epoch+1}, Task {task}, Loss={avg_loss_train:.4f}, Eval={eval_train:.4f}...")

        # === VALIDATION ===
        model.eval()
        total_loss_valid = 0.0
        predictions, ground_truths = [], []
        with torch.no_grad():
            for features, targets in valid_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                if task != 'cognitive_status':
                    outputs = outputs.squeeze(1)

                loss = criterion(outputs, targets)

                total_loss_valid += loss.item()
                if task == 'cognitive_status':
                    outputs_extend = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                elif task == 'sex':
                    outputs_extend = (outputs.detach().cpu().numpy() > 0.5).astype(int)
                else:
                    outputs_extend = outputs.detach().cpu().numpy()
                predictions.extend(outputs_extend)
                ground_truths.extend(targets.cpu().numpy())

            avg_loss_valid = total_loss_valid / len(valid_loader)

            if task in ['sex', 'cognitive_status']:
                eval_valid = accuracy_score(ground_truths, predictions)
            else:
                eval_valid = mean_absolute_error(ground_truths, predictions)
                # eval_valid = (abs(predictions - ground_truths) / ground_truths).mean()
        print(f"==VALID== Epoch {epoch+1}, Task {task}, Loss={avg_loss_valid:.4f}, Eval={eval_valid:.4f}...")
        # === METRIC ===
        writer.add_scalars(f'Loss/1-total_loss_{task}', {'train': avg_loss_train, 'test': avg_loss_valid}, epoch)
        writer.add_scalars(f'Loss/2-metric_{task}', {'train': eval_train, 'test': eval_valid}, epoch)
        
        # Check if this is the best model
        is_best = False
        if task == 'age':
            # For age (MAE), lower is better
            if eval_valid < best_valid_metric:
                is_best = True
                best_valid_metric = eval_valid
        else:
            # For sex and cognitive_status (accuracy), higher is better
            if eval_valid > best_valid_metric:
                is_best = True
                best_valid_metric = eval_valid
        
        if is_best:
            best_epoch = epoch + 1
            best_train_loss = avg_loss_train
            best_train_metric = eval_train
            best_valid_loss = avg_loss_valid
            best_valid_metric_value = eval_valid
            best_model_state = model.state_dict().copy()
            print(f"*** NEW BEST MODEL at Epoch {best_epoch} ***")
        
        # confusion matrix
        if task in ['sex', 'cognitive_status']:
            class_names = ['Female', 'Male'] if task == 'sex' else ['Normal', 'MCI', 'AD']
            cm_title = f"{task.capitalize()} Ep{epoch+1} (Acc {eval_valid:.2f})"
            cm_path = f"{output_dir}/confusion_matrix/{task}_epoch_{epoch+1}.png"

            save_confusion_matrix(
                y_true=ground_truths,
                y_pred=predictions,
                labels=class_names,
                title=cm_title,
                save_path=cm_path,
                normalize='true'
            )

    writer.close()
    
    # Print best model metrics
    print("\n" + "="*60)
    print("BEST MODEL SUMMARY")
    print("="*60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Metrics:")
    print(f"  - Loss: {best_train_loss:.4f}")
    metric_name = "Accuracy" if task in ['sex', 'cognitive_status'] else "MAE"
    print(f"  - {metric_name}: {best_train_metric:.4f}")
    print(f"Validation Metrics:")
    print(f"  - Loss: {best_valid_loss:.4f}")
    print(f"  - {metric_name}: {best_valid_metric_value:.4f}")
    print("="*60 + "\n")
    
    # Optionally save best model
    if best_model_state is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = f"{output_dir}/best_model_{task}.pth"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'train_loss': best_train_loss,
            'train_metric': best_train_metric,
            'valid_loss': best_valid_loss,
            'valid_metric': best_valid_metric_value,
            'task': task
        }, best_model_path)
        print(f"Best model saved to: {best_model_path}\n")


def evaluate_model(model, data_loader, task, device):
    """
    Evaluate a trained model on a given DataLoader.

    Args:
        model: The trained PyTorch model.
        data_loader: DataLoader for the evaluation set.
        task: Task type - "age", "sex", or "cognitive_status".
        device: Device to perform computation on.

    Returns:
        A dictionary containing loss and metric (MAE or Accuracy), plus predictions and ground_truths.
    """
    model.eval()
    criterion = None
    if task == 'age':
        criterion = nn.L1Loss()
    elif task == 'sex':
        criterion = nn.BCELoss()
    elif task == 'cognitive_status':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Invalid task value: {task}.')

    predictions, ground_truths = [], []

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            if task != 'cognitive_status':
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)

            # Process predictions
            if task == 'cognitive_status':
                out_np = torch.argmax(outputs, dim=1).cpu().numpy()
            elif task == 'sex':
                out_np = (outputs.cpu().numpy() > 0.5).astype(int)
            else:  # 'age'
                out_np = outputs.cpu().numpy()
            predictions.extend(out_np)
            ground_truths.extend(targets.cpu().numpy())

    if task == 'age':
        metric = mean_absolute_error(ground_truths, predictions)
    else:
        metric = accuracy_score(ground_truths, predictions)

    # Print out evaluation results similar to end of training
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    metric_name = "Accuracy" if task in ['sex', 'cognitive_status'] else "MAE"
    print(f"  - {metric_name}: {metric:.4f}")
    print("="*60 + "\n")

    # return {
    #     'loss': avg_loss,
    #     'metric': metric,
    #     'predictions': predictions,
    #     'ground_truths': ground_truths
    # }


def value_check(config):
    task = config['task']
    if task not in ['age', 'sex', 'cognitive_status']:
        raise ValueError(f'Unknown task: {task}. ' + \
                        'Choose one from "age", "sex", "cognitive_status".')
    

def save_confusion_matrix(y_true, y_pred, labels, title, save_path, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format=".2f" if normalize else "d")
    if normalize:
        disp.im_.set_clim(0, 1) 
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
