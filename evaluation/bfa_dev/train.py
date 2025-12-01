import yaml
import torch
import pathlib
from torch.utils.data import DataLoader
from models.network import BFANet
from data.dataset import BFADataset
from utils.trainer import train_model, value_check, evaluate_model

bfa_dir = pathlib.Path(__file__).parent

def main(config):
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() and config["gpu_id"] is not None else 'cpu')

    train_csv_path = str((bfa_dir / config['train_csv_path']).resolve())
    valid_csv_path = str((bfa_dir / config['valid_csv_path']).resolve())
    eval_csv_path = str((bfa_dir / config['eval_csv_path']).resolve())
    json_dir = str((bfa_dir / config['json_dir']).resolve())

    train_dataset = BFADataset(train_csv_path, json_dir, task=config['task'])
    valid_dataset = BFADataset(valid_csv_path, json_dir, task=config['task'])
    eval_dataset = BFADataset(eval_csv_path, json_dir, task=config['task'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # Use full dataset batch size for validation and evaluation to get accurate metrics
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    model = BFANet(input_dim=config['features'], hidden_dim=config['hidden_dim'], reduce_hidden_dim=config['reduce_hidden_dim'], task=config['task'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_model(model, train_loader, valid_loader, task=config['task'], 
                optimizer=optimizer, num_epochs=config['num_epochs'], device=device, 
                output_dir=config['output_dir'])
    evaluate_model(model, eval_loader, task=config['task'], device=device)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    value_check(config)
    main(config)