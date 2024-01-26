import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, load_checkpoint


torch.manual_seed(0)

# Remove annoying logs from the PIL library
logging.getLogger("PIL.TiffImagePlugin").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("PIL.Image").propagate = False

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), autocast(enabled=self.args.fp16_precision):
            for images in tqdm(val_loader, desc="Validation"):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                features = self.model(images)
                logits, labels = self.info_nce_loss(features)  # Update this line if needed
                loss = self.criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = correct / total
        val_loss /= len(val_loader)

        self.model.train()
        return val_loss, val_accuracy

    def train(self, train_loader, val_loader=None):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter, start_epochs, end_epochs, best_val_loss = 0, 0, self.args.epochs, np.inf

        if self.args.ckpt:
            self.model, self.optimizer, self.scheduler, start_epochs = load_checkpoint(self.model, self.optimizer,
                                                                                       self.scheduler, self.args.ckpt)
            end_epochs += start_epochs


        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {(not self.args.disable_cuda) and torch.cuda.is_available()}.")

        for epoch_counter in range(start_epochs, end_epochs):
            #for images, _ in tqdm(train_loader):

            # YO:: OADS dataloader only returns one object
            for images in tqdm(train_loader):

                # YO:: not sure if this is needed
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            # Validation on val dataset every x epochs
            if epoch_counter >= 0 and epoch_counter % self.args.validation_interval == 0 and val_loader is not None:
                # Perform validation using info_nce_loss
                val_loss, val_accuracy = self.validate(val_loader)
                is_best = False

                # Log and save results
                self.writer.add_scalar('val_loss', val_loss, global_step=n_iter)
                self.writer.add_scalar('is_best', is_best, global_step=n_iter)
                self.writer.add_scalar('val_accuracy', val_accuracy, global_step=n_iter)

                # Save checkpoint if the model performs better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, f"best_checkpoint.pth.tar"))

                logging.debug(f"\t\tValidation loss: {val_loss}\t Validation accuracy: {val_accuracy}"
                              f"\tis best (so far): {is_best}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(end_epochs)
        save_checkpoint({
            'epoch': end_epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
