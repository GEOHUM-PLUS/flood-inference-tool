import sys
sys.path.insert(1, '/media/bruno/Matosak/repos/SenForFlood')

import torch
from torch import nn
from SenForFlood import SenForFlood
from tqdm.auto import tqdm
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import argparse
import rasterio as r
import glob
from scipy import stats

from models import SimpleUNet, SimpleUNetEmb
from helpers import get_points_and_distance_map, get_points_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE)

class SenForFlood_distance_maps(torch.utils.data.Dataset):
    def __init__(self, events=None, chip_size=256, number_of_points_map=100, number_of_points_loss=500):
        super().__init__()
        data_to_include = ['s1_during_flood', 'terrain', 'SatCLIP_embedding', 'flood_mask_v1.1']
        self.senforflood = SenForFlood(
            dataset_folder='/media/bruno/Matosak/SenForFlood',
            chip_size=chip_size,
            data_to_include=data_to_include,
            percentile_scale_bttm=5,
            percentile_scale_top=95,
            events=events,
            use_data_augmentation=True
        )
        self.number_of_points_map = number_of_points_map
        self.number_of_points_loss = number_of_points_loss
    
    def __len__(self):
        return len(self.senforflood)
    
    def __getitem__(self, index):
        # getting EO data
        s1df, t, emb, fm = self.senforflood[index]

        # generating the rest
        cf, cn, dmap = get_points_and_distance_map(
            s1df, t, p=0.3,
            max_points_per_class_map=self.number_of_points_map, 
            max_points_per_class_loss=self.number_of_points_loss
        )
        dmap = torch.Tensor(np.array(dmap))

        cf, cn = get_points_loss(
            s1df, t, fm, p=0.3,
            max_points_per_class_loss=self.number_of_points_loss
        )

        samples_ind = [
            np.concatenate([cf[0], cn[0]]),
            np.concatenate([cf[1], cn[1]]),
            np.concatenate([[1]*len(cf[0]), [0]*len(cn[0])])
        ]

        return s1df, t, emb, dmap, samples_ind

def collate(batch):
    s1df = torch.stack([item[0] for item in batch])
    t = torch.stack([item[1] for item in batch])
    emb = torch.stack([item[2] for item in batch])
    dmap = torch.stack([item[3] for item in batch])
    samples_inds = [item[4] for item in batch]

    return s1df, t, emb, dmap, samples_inds
    
def train_loop(model, dataloader, optimizer, loss_fn, epochs, model_folder, save_epochs_intervals=5, starting_epoch=0, losses=[]):
    os.makedirs(model_folder, exist_ok=True)
    step = 0
    times = []
    for epoch in range(starting_epoch, epochs, 1):
        t = time.time()
        model.train()
        # optimizer.zero_grad()
        for ind, (s1df, terr, emb, dmap, si) in enumerate(dataloader):
            optimizer.zero_grad()

            samples_ind = [
                np.concatenate([[i]*len(si[i][0]) for i in range(len(si))]),
                np.concatenate([si[i][0] for i in range(len(si))]),
                np.concatenate([si[i][1] for i in range(len(si))]),
                np.concatenate([si[i][2] for i in range(len(si))]),
            ]

            x0 = torch.cat([s1df[:,:2], terr[:,1][:,None,:,:], dmap], axis=1).to(DEVICE)

            if 'Emb' in model.name:
                x1_ = model(x0, emb.to(torch.float32).to(DEVICE))
            else:
                x1_ = model(x0)

            x1 = nn.functional.one_hot(torch.Tensor(samples_ind[-1]).to(torch.long)).to(torch.float32).to(DEVICE)
            loss = loss_fn(x1_[samples_ind[0], :, samples_ind[1], samples_ind[2]], x1)

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            times.append(time.time()-t)
            step+=1
            if step%50==0:
                plot_loss(f'{model_folder}/losses.png', losses)
            t = time.time()
            print(f'Epoch: {epoch+1} Step: {ind+1}/{len(dataloader)} Loss: {f"{losses[-1]:.4f}" if ind!=len(dataloader)-1 else f"{np.mean(losses[-len(dataloader):]):.4f}"} {np.mean(times[max(-10, -len(times)):]):.1f}s/i{f"  rt: {str(datetime.timedelta(seconds=int(np.mean(times[max(-10, -len(times))])*(len(dataloader)-1-ind))))}" if ind!=(len(dataloader)-1) else f"  total_time: {str(datetime.timedelta(seconds=int(np.sum(times[-len(dataloader):]))))}"}', end='\r' if ind!=len(dataloader)-1 else '\n')
        
        # do some things by the end of an epoch
        if (epoch+1)%save_epochs_intervals==0 or (epoch+1)==epochs:
            plot_loss(f'{model_folder}/losses.png', losses)
            model.eval()
            with torch.no_grad():
                # saving checkpoints
                os.makedirs(f'{model_folder}/checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'model_name': model.name,
                    'chip_size': int(s1df.shape[-1]),
                    'use_emb': True
                    }, f'{model_folder}/checkpoints/model-{epoch+1:04d}.pt')
                
                # plotting examples
                os.makedirs(f'{model_folder}/plots/epoch-{epoch+1:04d}', exist_ok=True)
                for i in range(8):
                    plot_inference_example(
                        s1df[i,:3].moveaxis(0,-1),
                        terr[i,1], 
                        torch.argmax(x1_[i].detach().cpu(), dim=0), 
                        f'{model_folder}/plots/epoch-{epoch+1:04d}/sample-{epoch+1:03d}-{i}.png')

def plot_loss(save_path, losses):
    series_np = np.asarray(losses)
    series_smooth = np.copy(series_np)
    overlap = 30
    for i in range(len(series_smooth)):
        series_smooth[i] = np.median(series_np[max(-overlap+i,0):min(overlap+i, len(series_smooth))])
    
    plt.figure(figsize=(8,4))
    plt.plot(losses, alpha=0.5, label='raw loss')
    plt.plot(series_smooth, label='smoothed loss')
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_inference_example(s1, terr, inf, save_path):
    f, ax = plt.subplots(1,3, figsize=(6,2))

    ax[0].imshow(s1)
    ax[0].axis('off')

    ax[1].imshow(terr)
    ax[1].axis('off')

    ax[2].imshow(inf, vmin=0, vmax=1)
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_chip(s1, terr, dmap, result, name):
    f, ax = plt.subplots(2,2)

    ax[0,0].imshow(s1[0], vmin=0, vmax=1)
    ax[0,1].imshow(terr[0], vmin=0, vmax=1)
    ax[1,0].imshow(dmap[0], vmin=0, vmax=1)
    ax[1,1].imshow(result, vmin=0, vmax=1)

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    plt.tight_layout()

    plt.savefig(name)
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-id', '--id', type=str, required=True)
    parser.add_argument('-emb', '--use-embeddings', action='store_true')
    parser.add_argument('-cs', '--chip-size', type=int, default=256)
    parser.add_argument('-pm', '--points-map', type=int, default=100)
    parser.add_argument('-pl', '--points-loss', type=int, default=500)
    parser.add_argument('-e', '--max-epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)

    args = parser.parse_args()

    dataset = SenForFlood_distance_maps(
        number_of_points_map=args.points_map,
        number_of_points_loss=args.points_loss,
        chip_size=args.chip_size,
        # events=[
        #     'DFO_4459_Bangladesh', 'DFO_4484_Bangladesh', 'DFO_4508_Bangladesh',
        #     'DFO_4769_Bangladesh', 'DFO_4939_Bangladesh', 'DFO_5106_Bangladesh',
        #     'DFO_4466_Bangladesh'
        # ]
    )
    print(f'Total Samples: {len(dataset):,}')

    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate, num_workers=6)

    if args.use_embeddings:
        model = SimpleUNetEmb(4, 2, args.chip_size, 0.2).to(DEVICE)
    else:
        model = SimpleUNet(4, 2, 0.2).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    starting_epoch = 0
    losses = []

    if glob.glob(f'models_trainings/{args.id}/checkpoints/model-*.pt'):
        models_paths = glob.glob(f'models_trainings/{args.id}/checkpoints/model-*.pt')
        models_paths.sort()
        data = torch.load(models_paths[-1], weights_only=True)

        model.load_state_dict(data['model_state_dict'])
        optim.load_state_dict(data['optimizer_state_dict'])
        starting_epoch = data['epoch']+1
        losses = data["losses"]
    else:
        os.makedirs(f'models_trainings/{args.id}', exist_ok=True)
    
    # saving parameters before starting training
    with open(f'models_trainings/{args.id}/hyperparam.txt', 'w') as f:
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write(f'{key}: {args_dict[key]}\n')

    train_loop(
        model=model,
        dataloader=dl,
        optimizer=optim,
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs = args.max_epochs,
        save_epochs_intervals=1,
        model_folder=f'models_trainings/{args.id}',
        starting_epoch = starting_epoch,
        losses = losses
    )