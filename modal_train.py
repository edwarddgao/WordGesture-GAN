#!/usr/bin/env python3
"""
WordGesture-GAN Training on Modal with Checkpointing

Usage:
    python modal_train.py                    # Train 200 epochs (resumes if checkpoint exists)
    python modal_train.py --epochs 50        # Train 50 epochs
    python modal_train.py --eval-only        # Just run evaluation on saved model
    python modal_train.py --no-resume        # Start fresh, ignore existing checkpoint
"""

import os
if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('wordgesture-gan')
volume = modal.Volume.from_name('wordgesture-data', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'wandb', 'pillow')
)

WANDB_KEY = 'd68f9b4406a518b2095a579d37b0355bc18ad1a8'


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=7200)
def train(num_epochs: int = 200, resume: bool = True, checkpoint_every: int = 10):
    """Train WordGesture-GAN with checkpointing to Modal Volume."""
    import sys
    sys.path.insert(0, '/data')

    import torch
    import numpy as np
    import random
    import wandb
    from pathlib import Path
    from datetime import datetime

    from src.config import ModelConfig, TrainingConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
    from src.models import Generator, Discriminator, VariationalEncoder as Encoder

    device = 'cuda'
    checkpoint_dir = Path('/data/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Training for {num_epochs} epochs, checkpoints every {checkpoint_every}')

    # Config
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    training_config = TrainingConfig(batch_size=512, num_epochs=num_epochs, n_critic=5)

    # Data
    keyboard = QWERTYKeyboard()
    gestures, protos = load_dataset_from_zip('/data/swipelogs.zip', keyboard, model_config, training_config)
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=seed)
    train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=512, num_workers=2)
    print(f'Data: {len(train_ds)} train, {len(test_ds)} test')

    # Models
    generator = Generator(model_config).to(device)
    discriminator_1 = Discriminator(model_config).to(device)
    discriminator_2 = Discriminator(model_config).to(device)
    encoder = Encoder(model_config).to(device)

    # Optimizers
    lr = 0.0002
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D1 = torch.optim.Adam(discriminator_1.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(discriminator_2.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = checkpoint_dir / 'latest.pt'
    if resume and checkpoint_path.exists():
        print(f'Loading checkpoint from {checkpoint_path}...')
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        discriminator_1.load_state_dict(ckpt['discriminator_1'])
        discriminator_2.load_state_dict(ckpt['discriminator_2'])
        encoder.load_state_dict(ckpt['encoder'])
        optimizer_G.load_state_dict(ckpt['optimizer_G'])
        optimizer_D1.load_state_dict(ckpt['optimizer_D1'])
        optimizer_D2.load_state_dict(ckpt['optimizer_D2'])
        optimizer_E.load_state_dict(ckpt['optimizer_E'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    if start_epoch >= num_epochs:
        print(f'Already trained to epoch {start_epoch}, nothing to do.')
        return {'status': 'already_trained', 'epoch': start_epoch}

    # wandb
    os.environ['WANDB_API_KEY'] = WANDB_KEY
    run = wandb.init(
        project='wordgesture-gan',
        name=f'train-{datetime.now().strftime("%Y%m%d-%H%M")}',
        config={'epochs': num_epochs, 'batch_size': 512, 'lr': lr, 'latent_dim': 32},
        resume='allow' if resume else False
    )

    # Loss weights
    lambda_rec, lambda_lat, lambda_kld = 5.0, 0.5, 0.05

    print(f'Training epochs {start_epoch} to {num_epochs-1}...')

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        encoder.train()

        epoch_d1, epoch_d2, epoch_rec, n_batches = 0, 0, 0, 0

        for batch in train_loader:
            real = batch['gesture'].to(device)
            proto = batch['prototype'].to(device)
            bs = real.size(0)

            # === Cycle 1: z -> X' -> z' ===
            for _ in range(5):
                optimizer_D1.zero_grad()
                z = torch.randn(bs, 32, device=device)
                with torch.no_grad():
                    fake = generator(proto, z)
                d1_loss = -discriminator_1(real).mean() + discriminator_1(fake).mean()
                d1_loss.backward()
                optimizer_D1.step()
                for p in discriminator_1.parameters():
                    p.data.clamp_(-0.01, 0.01)

            optimizer_G.zero_grad()
            z = torch.randn(bs, 32, device=device)
            fake = generator(proto, z)
            g1_loss = -discriminator_1(fake).mean()
            z_rec, _, _ = encoder(fake)
            lat_loss = torch.mean(torch.abs(z - z_rec))
            (g1_loss + lambda_lat * lat_loss).backward()
            optimizer_G.step()

            # === Cycle 2: X -> z -> X' ===
            for _ in range(5):
                optimizer_D2.zero_grad()
                with torch.no_grad():
                    z_enc, _, _ = encoder(real)
                    fake2 = generator(proto, z_enc)
                d2_loss = -discriminator_2(real).mean() + discriminator_2(fake2).mean()
                d2_loss.backward()
                optimizer_D2.step()
                for p in discriminator_2.parameters():
                    p.data.clamp_(-0.01, 0.01)

            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            z_enc, mu, logvar = encoder(real)
            fake2 = generator(proto, z_enc)
            g2_loss = -discriminator_2(fake2).mean()
            rec_loss = torch.mean((real - fake2) ** 2)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            (g2_loss + lambda_rec * rec_loss + lambda_kld * kld_loss).backward()
            optimizer_G.step()
            optimizer_E.step()

            epoch_d1 += d1_loss.item()
            epoch_d2 += d2_loss.item()
            epoch_rec += rec_loss.item()
            n_batches += 1

        wandb.log({'epoch': epoch, 'd1': epoch_d1/n_batches, 'd2': epoch_d2/n_batches, 'rec': epoch_rec/n_batches})
        print(f'Epoch {epoch+1}/{num_epochs} - D1:{epoch_d1/n_batches:.3f} D2:{epoch_d2/n_batches:.3f} rec:{epoch_rec/n_batches:.4f}')

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            ckpt = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator_1': discriminator_1.state_dict(),
                'discriminator_2': discriminator_2.state_dict(),
                'encoder': encoder.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D1': optimizer_D1.state_dict(),
                'optimizer_D2': optimizer_D2.state_dict(),
                'optimizer_E': optimizer_E.state_dict(),
            }
            torch.save(ckpt, checkpoint_dir / 'latest.pt')
            torch.save(ckpt, checkpoint_dir / f'epoch_{epoch+1}.pt')
            volume.commit()
            print(f'Checkpoint saved at epoch {epoch+1}')

    wandb.finish()
    return {'status': 'complete', 'final_epoch': num_epochs}


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=1800)
def evaluate():
    """Evaluate trained model with all paper metrics."""
    import sys
    sys.path.insert(0, '/data')

    import torch
    import numpy as np
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import savgol_filter

    from src.config import ModelConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split
    from src.models import Generator
    from src.config import TrainingConfig

    device = 'cuda'
    checkpoint_path = Path('/data/checkpoints/latest.pt')

    if not checkpoint_path.exists():
        return {'error': 'No checkpoint found. Run train() first.'}

    print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load model
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    generator = Generator(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    epoch = ckpt['epoch'] + 1
    print(f'Loaded checkpoint from epoch {epoch}')

    # Load test data
    keyboard = QWERTYKeyboard()
    training_config = TrainingConfig()
    gestures, protos = load_dataset_from_zip('/data/swipelogs.zip', keyboard, model_config, training_config)
    _, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=42)

    # Generate
    n = min(1000, len(test_ds))
    real_g, fake_g = [], []
    with torch.no_grad():
        for i in range(n):
            item = test_ds[i]
            proto = item['prototype'].unsqueeze(0).to(device)
            z = torch.randn(1, 32, device=device)
            fake = generator(proto, z).cpu().numpy()[0]
            real_g.append(item['gesture'].numpy())
            fake_g.append(fake)
    real_g, fake_g = np.array(real_g), np.array(fake_g)

    # L2 Wasserstein
    dist = np.array([[np.sqrt(np.sum((real_g[i,:,:2] - fake_g[j,:,:2])**2)) for j in range(n)] for i in range(n)])
    r, c = linear_sum_assignment(dist)
    l2_xy = dist[r, c].mean()

    # DTW
    def dtw(a, b):
        n, m = len(a), len(b)
        d = np.full((n+1, m+1), np.inf)
        d[0,0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.sqrt(np.sum((a[i-1,:2] - b[j-1,:2])**2))
                d[i,j] = cost + min(d[i-1,j], d[i,j-1], d[i-1,j-1])
        return d[n,m]
    dtw_dist = np.array([[dtw(real_g[i], fake_g[j]) for j in range(n)] for i in range(n)])
    r2, c2 = linear_sum_assignment(dtw_dist)
    dtw_xy = dtw_dist[r2, c2].mean()

    # Jerk
    def jerk(gs):
        j = [np.mean(np.abs(savgol_filter(g[:,:2].flatten(), 5, 3, deriv=3))) for g in gs if len(g)>=5]
        return np.mean(j)
    jerk_real, jerk_fake = jerk(real_g), jerk(fake_g)

    # Velocity correlation
    vcorrs = []
    for i in range(n):
        vr = np.diff(real_g[i,:,:2], axis=0).flatten()
        vf = np.diff(fake_g[c[i],:,:2], axis=0).flatten()
        if len(vr) == len(vf):
            cr = np.corrcoef(vr, vf)[0,1]
            if not np.isnan(cr):
                vcorrs.append(cr)
    vcorr = np.mean(vcorrs)

    # Precision/Recall (k=3)
    def knn_r(data, k=3):
        return [sorted([np.sqrt(np.sum((data[i,:,:2]-data[j,:,:2])**2)) for j in range(len(data)) if i!=j])[k-1] for i in range(len(data))]
    rr, fr = knn_r(real_g), knn_r(fake_g)
    prec = sum(1 for i in range(n) if any(np.sqrt(np.sum((fake_g[i,:,:2]-real_g[j,:,:2])**2)) <= rr[j] for j in range(n))) / n
    rec = sum(1 for i in range(n) if any(np.sqrt(np.sum((real_g[i,:,:2]-fake_g[j,:,:2])**2)) <= fr[j] for j in range(n))) / n

    print('\n' + '='*65)
    print(f'{"RESULTS (epoch "+str(epoch)+")":<30} {"Paper":<12} {"Ours":<12}')
    print('='*65)
    print(f'{"L2 Wasserstein (x,y)":<30} {4.409:<12.3f} {l2_xy:<12.3f}')
    print(f'{"DTW Wasserstein (x,y)":<30} {2.146:<12.3f} {dtw_xy:<12.3f}')
    print(f'{"Jerk (generated)":<30} {0.0058:<12.4f} {jerk_fake:<12.4f}')
    print(f'{"Velocity Correlation":<30} {0.40:<12.2f} {vcorr:<12.2f}')
    print(f'{"Precision":<30} {0.973:<12.3f} {prec:<12.3f}')
    print(f'{"Recall":<30} {0.258:<12.3f} {rec:<12.3f}')
    print('='*65)

    return {
        'epoch': int(epoch), 'l2_xy': float(l2_xy), 'dtw_xy': float(dtw_xy),
        'jerk_fake': float(jerk_fake), 'vcorr': float(vcorr),
        'precision': float(prec), 'recall': float(rec)
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    async with app.run():
        if args.eval_only:
            result = await evaluate.remote.aio()
        else:
            result = await train.remote.aio(num_epochs=args.epochs, resume=not args.no_resume)
            if result.get('status') == 'complete':
                print('\nRunning evaluation...')
                result['eval'] = await evaluate.remote.aio()
    print(f'\nResult: {result}')


if __name__ == '__main__':
    asyncio.run(main())
