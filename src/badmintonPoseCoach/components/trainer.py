import torch
from torch.optim.lr_scheduler import StepLR
from badmintonPoseCoach.entity.config_entity import TrainingConfig
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

class Trainer:
    def __init__(self, config: TrainingConfig,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,):
        self.config = config
        self.device = config.params_device

        self.model = torch.load(self.config.updated_base_model_path, weights_only=False).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.params_lr)
        self.scheduler = StepLR(self.optimizer, step_size=config.params_step_size, gamma=config.params_gamma)

        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

         # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.params_use_amp and self.device == "cuda")
        # Checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)


    def save_model(self):
        torch.save(self.model, self.config.trained_model_path)

    def _step_batch(self, batch, train: bool = True) -> tuple[float, float]:
        packed = batch["packed"]
        labels = batch["labels"].to(self.device)
        packed = packed.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):

            logits = self.model(packed)

            loss = self.criterion(logits, labels)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        return loss.item(), acc

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in tqdm(self.train_loader):
            loss, acc = self._step_batch(batch, train=True)
            total_loss += loss
            total_acc += acc
            n += 1
        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> tuple[float, float]:
        self.model.eval()
        loader = {"val": self.val_loader, "test": self.test_loader}[split]
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in tqdm(loader):
            loss, acc = self._step_batch(batch, train=False)
            total_loss += loss
            total_acc += acc
            n += 1
        return total_loss / max(n, 1), total_acc / max(n, 1)

    def fit(self):
        best_val_acc = 0.0
        for epoch in range(1, self.config.params_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate("val")
            self.scheduler.step()

            print(f"Epoch {epoch:03d}: train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f}, acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = Path(self.config.checkpoint_dir) / f"best.pkl"
                torch.save({
                    "model_state": self.model.state_dict(),
                    "cfg": self.config.__dict__,
                    "val_acc": val_acc,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print(f"Best val acc: {best_val_acc:.4f}")

    @torch.no_grad()
    def test(self):
        ckpt_path = Path(self.config.checkpoint_dir) / "best.pkl"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"])
            print(f"Loaded checkpoint from {ckpt_path}")
        test_loss, test_acc = self.evaluate("test")
        print(f"Test: loss {test_loss:.4f}, acc {test_acc:.4f}")
