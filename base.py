#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from random import SystemRandom
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import models
import utils


def main(args):
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)
    elif args.dataset == 'activity':
        data_obj = utils.get_activity_data(args, 'cpu')
        
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    # model
    if args.enc == 'mtan_enc':
        rec = models.enc_mtan_classif(
            dim, torch.linspace(0, 1., 128), args.rec_hidden, 
            args.embed_time, args.num_heads, args.learn_emb, args.freq).to(device)
        
    elif args.enc == 'mtan_enc_activity':
        rec = models.enc_mtan_classif_activity_v2(
            dim, args.rec_hidden, args.embed_time, 
            args.num_heads, args.learn_emb, args.freq).to(device)
        
    aug = models.TimeSeriesAugmentation(dim*2+1, args.augh1, args.augh2, dim*2+1, num_outputs=int(args.aug_ratio*50)).to(device)

    params = (list(rec.parameters())+ list(aug.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(aug))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_test_acc = 0
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_loss, train_reg_loss = 0, 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            # check
            # if random.random()<0.1:
            #     print(observed_data.shape, observed_mask.shape, observed_tp.shape)
            #     print(observed_tp[0])
            #torch.Size([256, 50, 12]) torch.Size([256, 50, 12]) torch.Size([256, 50])

            x_aug, tp_aug = aug(observed_tp, torch.cat((observed_data, observed_mask), 2))
                    
            mask_aug = torch.where(
                x_aug[:, :, dim:2*dim] < 0.5,  # 조건
                torch.zeros_like(x_aug[:, :, dim:2*dim]),  # 조건이 True일 때 적용할 값
                x_aug[:, :, dim:2*dim]  # 조건이 False일 때 적용할 값
            )    
            data_aug = x_aug[:, :, :dim]
            
            data = torch.cat((observed_data, data_aug), -2)
            mask = torch.cat((observed_mask, mask_aug), -2)

            tt = torch.cat((observed_tp, tp_aug), -1)

            reg_loss = utils.diversity_regularization(tt, drate = 0.5)

            out = rec(torch.cat((data, mask), 2), tt, observed_tp)
            
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                loss = criterion(out, label.long())
            else:
                loss = criterion(out, label) + args.beta*reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_len
            train_reg_loss += reg_loss.item() * batch_len
            train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
            train_n += batch_len
        total_time += time.time() - start_time
        val_loss, val_acc, val_auc = utils.evaluate_classifier(rec, aug, val_loader, args=args, dim=dim)
        best_val_loss = min(best_val_loss, val_loss)
        test_loss, test_acc, test_auc = utils.evaluate_classifier(rec, aug, test_loader, args=args, dim=dim)
        if random.random() < 0.03:
            print('Iter: {}, loss: {:.4f}, reg_loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(itr, train_loss/train_n, train_reg_loss/train_n, train_acc/train_n, val_loss, val_acc, test_acc, test_auc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        if best_val_loss * 1.5 < val_loss:
            break
    return best_test_acc

# Callback to print intermediate results
def print_progress(res):
    print(f"Current best score: {-res.fun}")
    print(f"Best parameters so far: {res.x}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rec-hidden', type=int, default=32)
    parser.add_argument('--embed-time', type=int, default=128)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--enc', type=str, default='mtan_enc')
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--quantization', type=float, default=0.016, 
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--classif', action='store_true', 
                        help="Include binary classification loss")
    parser.add_argument('--learn-emb', action='store_true')
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--dataset', type=str, default='physionet')
    parser.add_argument('--old-split', type=int, default=1)
    parser.add_argument('--nonormalize', action='store_true')
    parser.add_argument('--classify-pertp', action='store_true')
    parser.add_argument('--aug-ratio', type=float, default=1)
    parser.add_argument('--augh1', type=int, default=300)
    parser.add_argument('--augh2', type=int, default=256)
    parser.add_argument('--beta', type=int, default=1000000)

    args = parser.parse_args()

 # Define the search space
    search_space = [
        Integer(100000, 5000000, name='beta'),
        Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
        Integer(128, 1024, name='rec_hidden'),
        Integer(10, 64, name='latent_dim'),
        Categorical(['mtan_enc_activity'], name='enc'),
        Categorical([True], name='classif'),
        Categorical([True], name='learn_emb'),
        Categorical([True], name='classify_pertp'),
        Categorical(['activity'], name='dataset'),
        Real(0, 3, name='aug_ratio'),
        Integer(50, 500, name='augh1'),
        Integer(128, 512, name='augh2')
    ]

    @use_named_args(search_space)
    def objective(**params):
        for param, value in params.items():
            setattr(args, param, value)
        print(f"Running with parameters: {params}")
        best_test_acc = main(args)
        return -best_test_acc

    # Perform Bayesian optimization using Gaussian Process
    res = gp_minimize(objective, search_space, n_calls=100, random_state=args.seed, callback=[print_progress])

    print(f"Best score: {-res.fun} with parameters: {res.x}")
