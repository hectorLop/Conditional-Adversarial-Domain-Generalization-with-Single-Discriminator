from typing import Dict, List, Tuple
import torch
import numpy as np
import argparse
from torch import nn
import yaml
import pandas as pd

from sklearn.metrics import roc_auc_score
from adversarial.adversarial import AdversarialNetwork, Classifier, Discriminator
from adversarial.dataset import (
    AdversarialDataset,
    get_transforms
)
from adversarial.config import Config
from adversarial.utils import (
    fix_all_seeds,
    freeze_unfreeze,
    get_ground_truth_vector
)
from torch.utils.data import DataLoader

def train_step(
    model : nn.Module,
    train_loader : DataLoader,
    config : Config,
    class_criterion : object,
    disc_criterion : object,
    extractor_criterion : object,
    optimizer : torch.optim.Optimizer
) -> Tuple[float, float, float, float]:
    model.train()

    class_loss_accum, disc_loss_accum, extr_loss_accum = 0., 0., 0.
    y_train = []
    preds = []

    for images, domains, labels in train_loader:
        images = images.to(config.DEVICE)
        domains = domains.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Set the gradients to zero before backprop step
        optimizer.zero_grad()
        
        # # # # # # # # # # # # # #
        # Step 1: Classification  #
        # # # # # # # # # # # # # #
        
        freeze_unfreeze(model.feature_extractor, True)
        freeze_unfreeze(model.discriminator, True)
        freeze_unfreeze(model.classifier, True)
        
        # Get predictions and calculate the loss
        y_preds_class = model(images)
        y_preds_class = y_preds_class.to(config.DEVICE)

        class_loss = class_criterion(y_preds_class.squeeze(), labels)
        class_loss_accum += class_loss.item()
        
        # Backward step
        class_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        y_train.append(labels.detach().cpu().numpy())
        preds.append(y_preds_class.softmax(1).detach().cpu().numpy())
        

        # # # # # # # # # # # # #
        # Step 2: Discriminator # 
        # # # # # # # # # # # # #
        
        freeze_unfreeze(model.feature_extractor, False)
        freeze_unfreeze(model.discriminator, True)
        freeze_unfreeze(model.classifier, True)
        
        # Get predictions and calculate the loss
        y_preds_disc = model.forward_disc(images)
        y_preds_disc = y_preds_disc.to(config.DEVICE)
        
        disc_loss = disc_criterion(y_preds_disc.squeeze(), domains)
        disc_loss_accum += disc_loss.item()
        
        # Backward step
        disc_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        # # # # # # # # # # #
        # Step 3: Extractor # 
        # # # # # # # # # # #
        
        freeze_unfreeze(model.feature_extractor, True)
        freeze_unfreeze(model.discriminator, False)
        freeze_unfreeze(model.classifier, True)
        
        # Get predictions and calculate the loss
        y_preds_extr = model.forward_disc(images)
        y_preds_extr = y_preds_extr.to(config.DEVICE)
        
        gt_vector = get_ground_truth_vector(labels, config.N_DOMAINS, config.N_CLASSES)
        gt_vector = gt_vector.to(config.DEVICE)

        extr_loss = extractor_criterion(y_preds_extr.squeeze(), gt_vector)
        extr_loss_accum += extr_loss.item()
        
        # Backward step
        extr_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    y_train = np.concatenate(y_train)
    preds = np.concatenate(preds)
    preds = preds[np.arange(len(preds)), preds.argmax(1)]
    auc = roc_auc_score(y_train, preds)

    return class_loss_accum, disc_loss_accum, extr_loss_accum, auc


def val_step(model : nn.Module, val_loader : DataLoader,
             config : Config, criterion : object) -> Tuple[float, float]:
    model.eval()

    preds = []
    epoch_loss = 0
    y_test = []

    with torch.no_grad():
        for images, domains, labels in val_loader:
            images = images.to(config.DEVICE)
            domains = domains.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            y_preds = model(images)
            y_preds = y_preds.to(config.DEVICE)

            loss = criterion(y_preds.squeeze(), labels)

            y_test.append(labels.cpu().numpy())
            preds.append(y_preds.softmax(1).cpu().numpy())
            epoch_loss += loss.item()
            
    y_test = np.concatenate(y_test)
    preds = np.concatenate(preds)
    preds = preds[np.arange(len(preds)), preds.argmax(1)]
    auc = roc_auc_score(y_test, preds)

    return epoch_loss, auc


def fit(
    model : nn.Module,
    train_loader : DataLoader,
    val_loader : DataLoader,
    config : Config,
    filepath : str
) -> Tuple[nn.Module, List[float], List[float]]:
    model = model.to(config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.LEARNING_RATE,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    

    # Criterions for each step
    class_criterion = torch.nn.CrossEntropyLoss()
    disc_criterion = torch.nn.CrossEntropyLoss()
    extr_criterion = torch.nn.MSELoss()
    
    n_batches, n_batches_val = len(train_loader), len(val_loader)

    best_loss = np.inf
    val_loss_accum, train_loss_accum = [], []

    with torch.cuda.device(config.DEVICE):
        
        for epoch in range(1, config.EPOCHS + 1):
            class_loss, disc_loss, extr_loss, train_auc = train_step(model,
                                                                     train_loader,
                                                                     config,
                                                                     class_criterion,
                                                                     disc_criterion,
                                                                     extr_criterion,
                                                                     optimizer)

            class_loss = class_loss / n_batches
            disc_loss = disc_loss / n_batches
            extr_loss = extr_loss / n_batches
            
            val_loss, val_auc = val_step(model,
                                         val_loader,
                                         config,
                                         class_criterion)
            
            val_loss = val_loss / n_batches_val


            prefix = f"[Epoch {epoch:2d} / {config.EPOCHS:2d}]"
            print(prefix)
            print(f"{prefix} Train Class loss: {class_loss:7.5f}. Train Disc Loss: {disc_loss:7.5f}. Train Extr Loss: {extr_loss:7.5f}")
            print(f"{prefix} Val Class loss: {val_loss:7.5f}")
            print(f"{prefix} Train AUC-ROC: {train_auc:7.5f}. Val AUC-ROC: {val_auc:7.5f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f'{prefix} Save Val loss: {val_loss:7.5f}')
                torch.save(model.state_dict(), filepath)
                
            print(prefix)

    return model, train_loss_accum, val_loss_accum

def get_loaders(df_train, df_val, config=Config):
    ds_train = AdversarialDataset(df_train, get_transforms(config, augment=True), config)
    dl_train = DataLoader(ds_train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

    ds_val = AdversarialDataset(df_val, get_transforms(config, augment=False), config)
    dl_val = DataLoader(ds_val,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)
    
    return dl_train, dl_val

def train(parameters : Dict):
    fix_all_seeds(3088)
    
    train = pd.read_csv(parameters['train_set'])
    val = pd.read_csv(parameters['val_set'])

    train_loader, val_loader = get_loaders(train, val)

    print('Getting the model') 
    classifier = Classifier(256, 2)
    discriminator = Discriminator(256, 0.5, Config.N_DOMAINS, Config.N_CLASSES)
    model = AdversarialNetwork(discriminator, classifier,
                               parameters['model_name'], 2048)
    
    print('TRAINING')
    model, train_loss, val_loss = fit(model,
                                      train_loader,
                                      val_loader,
                                      Config,
                                      parameters['checkpoint'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)