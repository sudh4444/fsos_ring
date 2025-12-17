import os
import argparse
import time
import logging
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from model import SisFallDataset, FallDetectionCNN, SISFALL_ROOT
from loss import SigmoidLoss


def split_dataset(dataset, ratios=(0.7, 0.2, 0.1), seed=42):
    total = len(dataset)
    train_len = int(total * ratios[0])
    val_len = int(total * ratios[1])
    test_len = total - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))


def evaluate(model, loader, device, criterion=None):
    model.eval()
    total, correct = 0, 0
    losses = []
    with torch.no_grad():
        non_blocking = True if getattr(loader, 'pin_memory', False) else False
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking).float()
            y = y.to(device, non_blocking=non_blocking).long()
            out = model(x)
            if criterion is not None:
                losses.append(criterion(out, y).item())

            probs = torch.sigmoid(out[:, 1] - out[:, 0])
            preds = (probs > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_loss, acc


def compute_train_accuracy(model, loader, device):
    """Compute training accuracy on the training dataset."""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        non_blocking = True if getattr(loader, 'pin_memory', False) else False
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking).float()
            y = y.to(device, non_blocking=non_blocking).long()
            out = model(x)
            probs = torch.sigmoid(out[:, 1] - out[:, 0])
            preds = (probs > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def main(args):
    # Device selection: prefer CUDA when available, allow override via --device
    if getattr(args, 'device', None) and args.device.lower() != 'auto':
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(0)
            print(f"Using CUDA device: {props.name} (sm_{props.major}{props.minor})")
        except Exception:
            print("CUDA device detected but query failed; proceeding with CUDA if available.")

    torch.manual_seed(42)

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('train')

    dataset = SisFallDataset(SISFALL_ROOT)
    train_set, val_set, test_set = split_dataset(dataset)

    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    logger.info(f"Dataset sizes — train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    model = FallDetectionCNN().to(device)
    criterion = SigmoidLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    total_batches = len(train_loader)
    
    # Metrics tracking
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        non_blocking = True if pin_memory else False
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=non_blocking).float()
            y = y.to(device, non_blocking=non_blocking).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss * x.size(0)
            running_loss += batch_loss

            if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == total_batches):
                avg_batch_loss = running_loss / (args.log_interval if batch_idx >= args.log_interval else batch_idx)
                lr = optimizer.param_groups[0].get('lr', 0.0)
                logger.info(f"Epoch {epoch} [{batch_idx}/{total_batches}] batch_loss: {batch_loss:.4f} avg_batch_loss: {avg_batch_loss:.4f} lr: {lr:.6f}")
                running_loss = 0.0

        epoch_loss = epoch_loss / len(train_loader.dataset)
        train_acc = compute_train_accuracy(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start
        
        # Track metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch}/{args.epochs} - train_loss: {epoch_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    logger.info(f"Test - loss: {test_loss:.4f} acc: {test_acc:.4f}")
    
    # Save metrics to CSV
    metrics_file = os.path.join(args.save_dir, 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        for i in range(len(metrics['epoch'])):
            row = {k: metrics[k][i] for k in metrics.keys()}
            writer.writerow(row)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(metrics['epoch'], metrics['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(metrics['epoch'], metrics['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plot_file = os.path.join(args.save_dir, 'metrics_plot.png')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Plot saved to {plot_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='auto', help="Device to use: 'auto', 'cpu', or 'cuda'")
    parser.add_argument('--log-interval', type=int, default=50, help='Batches between log messages')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
