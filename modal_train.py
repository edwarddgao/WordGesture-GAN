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

# Image with local src package included
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0', 'wandb', 'pillow', 'matplotlib', 'fastdtw', 'joblib')
    .add_local_python_source('src')  # Include local src package with latest code
)

WANDB_KEY = 'd68f9b4406a518b2095a579d37b0355bc18ad1a8'


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=7200)
def train(num_epochs: int = 200, resume: bool = True, checkpoint_every: int = 10):
    """Train WordGesture-GAN with checkpointing to Modal Volume."""
    # src package is included via image.add_local_python_source('src')

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

    # Loss weights (from paper)
    lambda_feat = 1.0   # Feature matching
    lambda_rec = 5.0    # Reconstruction (L1)
    lambda_lat = 0.5    # Latent recovery
    lambda_kld = 0.05   # KL divergence
    # Note: Paper uses spectral normalization only (no weight clipping, no GP)

    def get_features(disc, x):
        """Get intermediate features from discriminator for feature matching."""
        batch = x.size(0)
        x_flat = x.view(batch, -1)
        feats = []
        for layer in disc.layers:
            x_flat = torch.nn.functional.leaky_relu(layer(x_flat), 0.2)
            feats.append(x_flat)
        return feats

    def feature_matching_loss(real_feats, fake_feats):
        """L1 distance between real and fake features."""
        loss = 0.0
        for rf, ff in zip(real_feats, fake_feats):
            loss += torch.mean(torch.abs(ff - rf.detach()))
        return loss / len(real_feats)

    print(f'Training epochs {start_epoch} to {num_epochs-1}...')

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        encoder.train()
        discriminator_1.train()
        discriminator_2.train()

        epoch_d1, epoch_d2, epoch_g, epoch_rec, epoch_feat, n_batches = 0, 0, 0, 0, 0, 0

        for batch in train_loader:
            real = batch['gesture'].to(device)
            proto = batch['prototype'].to(device)
            bs = real.size(0)

            # === Cycle 1: z -> X' -> z' ===
            # Train discriminator n_critic times
            for _ in range(5):
                optimizer_D1.zero_grad()
                z = torch.randn(bs, 32, device=device)
                with torch.no_grad():
                    fake = generator(proto, z)
                d1_loss = discriminator_1(fake).mean() - discriminator_1(real).mean()
                d1_loss.backward()
                optimizer_D1.step()

            # Train generator
            optimizer_G.zero_grad()
            z = torch.randn(bs, 32, device=device)
            fake = generator(proto, z)

            # WGAN loss
            g1_loss = -discriminator_1(fake).mean()

            # Feature matching loss
            real_feats = get_features(discriminator_1, real)
            fake_feats = get_features(discriminator_1, fake)
            feat_loss = feature_matching_loss(real_feats, fake_feats)

            # Latent recovery loss - freeze encoder but allow gradient through to generator
            # Paper: "freeze the encoder when updating latent code loss"
            for p in encoder.parameters():
                p.requires_grad = False
            z_rec, _, _ = encoder(fake)  # NO detach - gradient flows to generator
            lat_loss = torch.mean(torch.abs(z - z_rec))
            for p in encoder.parameters():
                p.requires_grad = True

            total_g1 = g1_loss + lambda_feat * feat_loss + lambda_lat * lat_loss
            total_g1.backward()
            optimizer_G.step()

            # === Cycle 2: X -> z -> X' ===
            # Train discriminator n_critic times
            for _ in range(5):
                optimizer_D2.zero_grad()
                with torch.no_grad():
                    z_enc, _, _ = encoder(real)
                    fake2 = generator(proto, z_enc)
                d2_loss = discriminator_2(fake2).mean() - discriminator_2(real).mean()
                d2_loss.backward()
                optimizer_D2.step()

            # Train generator + encoder
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            z_enc, mu, logvar = encoder(real)
            fake2 = generator(proto, z_enc)

            # WGAN loss
            g2_loss = -discriminator_2(fake2).mean()

            # Feature matching loss
            real_feats2 = get_features(discriminator_2, real)
            fake_feats2 = get_features(discriminator_2, fake2)
            feat_loss2 = feature_matching_loss(real_feats2, fake_feats2)

            # L1 reconstruction loss (not MSE!)
            rec_loss = torch.mean(torch.abs(real - fake2))

            # KL divergence
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            total_g2 = g2_loss + lambda_feat * feat_loss2 + lambda_rec * rec_loss + lambda_kld * kld_loss
            total_g2.backward()
            optimizer_G.step()
            optimizer_E.step()

            epoch_d1 += d1_loss.item()
            epoch_d2 += d2_loss.item()
            epoch_g += (g1_loss.item() + g2_loss.item()) / 2
            epoch_rec += rec_loss.item()
            epoch_feat += (feat_loss.item() + feat_loss2.item()) / 2
            n_batches += 1

        wandb.log({
            'epoch': epoch,
            'd1': epoch_d1/n_batches,
            'd2': epoch_d2/n_batches,
            'g_loss': epoch_g/n_batches,
            'rec': epoch_rec/n_batches,
            'feat': epoch_feat/n_batches
        })
        print(f'Epoch {epoch+1}/{num_epochs} - D1:{epoch_d1/n_batches:.3f} D2:{epoch_d2/n_batches:.3f} rec:{epoch_rec/n_batches:.4f}')

        # Log gesture images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                # Get a few samples
                sample_batch = next(iter(train_loader))
                sample_real = sample_batch['gesture'][:4].to(device)
                sample_proto = sample_batch['prototype'][:4].to(device)
                z = torch.randn(4, 32, device=device)
                sample_fake = generator(sample_proto, z)

                # Plot gestures
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for i in range(4):
                    # Real
                    r = sample_real[i].cpu().numpy()
                    axes[0, i].plot(r[:, 0], r[:, 1], 'b-', linewidth=1)
                    axes[0, i].scatter(r[::10, 0], r[::10, 1], c='blue', s=10)
                    axes[0, i].set_title(f'Real {i+1}')
                    axes[0, i].set_xlim(-1, 1)
                    axes[0, i].set_ylim(-1, 1)
                    axes[0, i].invert_yaxis()
                    axes[0, i].set_aspect('equal')

                    # Fake
                    f = sample_fake[i].cpu().numpy()
                    axes[1, i].plot(f[:, 0], f[:, 1], 'r-', linewidth=1)
                    axes[1, i].scatter(f[::10, 0], f[::10, 1], c='red', s=10)
                    axes[1, i].set_title(f'Generated {i+1}')
                    axes[1, i].set_xlim(-1, 1)
                    axes[1, i].set_ylim(-1, 1)
                    axes[1, i].invert_yaxis()
                    axes[1, i].set_aspect('equal')

                plt.tight_layout()
                wandb.log({'gestures': wandb.Image(fig)})
                plt.close(fig)
            generator.train()

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


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=3600)
def evaluate(n_samples: int = 200, checkpoint_epoch: int = None, truncation: float = 0.05):
    """Evaluate trained model with all paper metrics.

    Uses Improved Precision and Recall Metric (Kynkäänniemi et al., 2019)
    with k=3 for manifold estimation.

    Args:
        n_samples: Number of samples for evaluation (default 200)
        checkpoint_epoch: Specific epoch checkpoint to use (default: latest)
        truncation: Latent code truncation (z * truncation), default 0.05 to match paper
    """
    # src package is included via image.add_local_python_source('src')

    def log(msg):
        """Print with immediate flush for Modal streaming."""
        print(msg, flush=True)

    import torch
    import numpy as np
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import savgol_filter
    import scipy.linalg

    from src.config import ModelConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split
    from src.models import Generator
    from src.config import TrainingConfig

    device = 'cuda'
    if checkpoint_epoch is not None:
        checkpoint_path = Path(f'/data/checkpoints/epoch_{checkpoint_epoch}.pt')
    else:
        checkpoint_path = Path('/data/checkpoints/latest.pt')

    if not checkpoint_path.exists():
        return {'error': 'No checkpoint found. Run train() first.'}

    log(f'[remote] GPU: {torch.cuda.get_device_name(0)}')
    log(f'[remote] Evaluating with {n_samples} samples...')

    # Load model
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    generator = Generator(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    epoch = ckpt['epoch'] + 1
    log(f'[remote] Loaded checkpoint from epoch {epoch}')

    # Load train and test data
    keyboard = QWERTYKeyboard()
    training_config = TrainingConfig()
    gestures, protos = load_dataset_from_zip('/data/swipelogs.zip', keyboard, model_config, training_config)
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=42)

    # Generate (use fewer samples to speed up DTW)
    n = min(n_samples, len(test_ds))
    log(f'[1/9] Generating {n} samples (truncation={truncation})...')
    real_g, fake_g = [], []
    with torch.no_grad():
        for i in range(n):
            item = test_ds[i]
            proto = item['prototype'].unsqueeze(0).to(device)
            # Apply truncation trick: z * truncation reduces diversity, increases quality
            z = torch.randn(1, 32, device=device) * truncation
            fake = generator(proto, z).cpu().numpy()[0]
            real_g.append(item['gesture'].numpy())
            fake_g.append(fake)
            if (i + 1) % 50 == 0:
                log(f'  Generated {i+1}/{n}')
    real_g, fake_g = np.array(real_g), np.array(fake_g)
    log(f'  Done generating.')

    # L2 Wasserstein (vectorized with cdist)
    log(f'[2/9] Computing L2 Wasserstein distance...')
    from scipy.spatial.distance import cdist
    real_flat_xy = real_g[:, :, :2].reshape(n, -1)
    fake_flat_xy = fake_g[:, :, :2].reshape(n, -1)
    dist = cdist(real_flat_xy, fake_flat_xy, metric='euclidean')
    r, c = linear_sum_assignment(dist)
    l2_xy = dist[r, c].mean()
    log(f'  L2 Wasserstein: {l2_xy:.3f}')

    # DTW using fastdtw with parallel computation
    # Normalize by path length to get average per-step distance (like paper)
    log(f'[3/9] Computing DTW distance ({n}x{n} pairs, parallelized)...')
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    from joblib import Parallel, delayed

    seq_len = real_g.shape[1]  # 128

    def compute_dtw_row(i, real_g, fake_g):
        """Compute DTW for one row."""
        row = np.zeros(len(fake_g))
        for j in range(len(fake_g)):
            distance, _ = fastdtw(real_g[i, :, :2], fake_g[j, :, :2], dist=euclidean)
            row[j] = distance
        return row

    # Parallel DTW computation using all CPU cores
    dtw_rows = Parallel(n_jobs=-1, verbose=1)(
        delayed(compute_dtw_row)(i, real_g, fake_g) for i in range(n)
    )
    dtw_dist = np.array(dtw_rows)
    r2, c2 = linear_sum_assignment(dtw_dist)
    dtw_raw = dtw_dist[r2, c2].mean()
    # Normalize by sqrt(seq_length) to match paper's scale
    # Paper DTW (2.146) < L2 (4.409), suggesting normalization
    dtw_xy = dtw_raw / np.sqrt(seq_len)
    log(f'  DTW Wasserstein: {dtw_xy:.3f} (raw: {dtw_raw:.3f})')

    # Jerk (third derivative magnitude, no time normalization)
    # Paper uses Savitzky-Golay filter with window_size=5, polynomial degree 3
    log(f'[4/9] Computing jerk...')
    def compute_gesture_jerk(g):
        """Compute jerk as third derivative magnitude."""
        x, y = g[:, 0], g[:, 1]
        if len(x) < 5:  # Need enough points for window_size=5
            return 0.0
        d3x = savgol_filter(x, 5, 3, deriv=3)
        d3y = savgol_filter(y, 5, 3, deriv=3)
        return np.mean(np.sqrt(d3x**2 + d3y**2))
    jerk_real = np.mean([compute_gesture_jerk(g) for g in real_g])
    jerk_fake = np.mean([compute_gesture_jerk(g) for g in fake_g])
    log(f'  Jerk: {jerk_fake:.6f} (real: {jerk_real:.6f})')

    # Velocity correlation
    # Compare directly (i to i) since each fake is generated for same prototype as real
    log(f'[5/9] Computing velocity correlation...')
    vcorrs = []
    for i in range(n):
        vr = np.diff(real_g[i,:,:2], axis=0).flatten()
        vf = np.diff(fake_g[i,:,:2], axis=0).flatten()  # Compare directly, not c[i]
        if len(vr) == len(vf):
            cr = np.corrcoef(vr, vf)[0,1]
            if not np.isnan(cr):
                vcorrs.append(cr)
    vcorr = np.mean(vcorrs)
    log(f'  Velocity correlation: {vcorr:.3f}')

    # Acceleration correlation (2nd derivative using Savitzky-Golay filter)
    # Use savgol_filter like jerk for smoother derivatives
    log(f'[6/9] Computing acceleration correlation...')
    acorrs = []
    for i in range(n):
        # Acceleration = 2nd derivative using savgol_filter (smoother than double diff)
        xr, yr = real_g[i, :, 0], real_g[i, :, 1]
        xf, yf = fake_g[i, :, 0], fake_g[i, :, 1]
        if len(xr) >= 5:
            # 2nd derivative with window=5, poly=3
            ax_r = savgol_filter(xr, 5, 3, deriv=2)
            ay_r = savgol_filter(yr, 5, 3, deriv=2)
            ax_f = savgol_filter(xf, 5, 3, deriv=2)
            ay_f = savgol_filter(yf, 5, 3, deriv=2)
            ar = np.concatenate([ax_r, ay_r])
            af = np.concatenate([ax_f, ay_f])
            cr = np.corrcoef(ar, af)[0, 1]
            if not np.isnan(cr):
                acorrs.append(cr)
    acorr = np.mean(acorrs) if acorrs else 0.0
    log(f'  Acceleration correlation: {acorr:.3f}')

    # Duration RMSE (compare predicted vs real gesture duration)
    # Each fake[i] is generated for the same prototype as real[i], so compare directly
    log(f'[7/9] Computing duration RMSE...')
    # Duration is normalized to [0, 1] range (last timestamp should be ~1.0)
    real_durations = np.array([g[-1, 2] for g in real_g])  # Last timestamp
    fake_durations = np.array([g[-1, 2] for g in fake_g])

    # Debug: show duration statistics
    log(f'  Real durations: min={real_durations.min():.3f}, max={real_durations.max():.3f}, mean={real_durations.mean():.3f}')
    log(f'  Fake durations: min={fake_durations.min():.3f}, max={fake_durations.max():.3f}, mean={fake_durations.mean():.3f}')

    # Compute RMSE in normalized units
    # Paper reports 1180.3ms with avg duration 1946.8ms, so relative error ≈ 60%
    # For normalized [0,1] times, equivalent RMSE would be ~0.6
    duration_rmse = np.sqrt(np.mean((real_durations - fake_durations)**2))
    # Convert to approximate ms assuming avg duration of 2000ms
    duration_rmse_ms = duration_rmse * 2000
    log(f'  Duration RMSE: {duration_rmse:.4f} (normalized), ~{duration_rmse_ms:.1f}ms (approx)')

    # FID Score (Frechet Inception Distance adapted for gestures)
    # Paper uses trained autoencoder for feature extraction (Section 4.3)
    log(f'[8/9] Computing FID score with autoencoder...')
    from src.models import AutoEncoder
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    # Train autoencoder on FULL training set (not just eval samples) for better features
    log(f'  Loading full training set for autoencoder ({len(train_ds)} samples)...')
    train_gestures = np.array([train_ds[i]['gesture'].numpy() for i in range(len(train_ds))])
    train_tensor = torch.tensor(train_gestures, dtype=torch.float32)

    autoencoder = AutoEncoder(model_config, hidden_dim=64).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    ae_criterion = torch.nn.L1Loss()

    # Create dataloader for training gestures
    ae_dataset = TensorDataset(train_tensor)
    ae_loader = TorchDataLoader(ae_dataset, batch_size=64, shuffle=True)

    log(f'  Training autoencoder on {len(train_ds)} gestures (50 epochs)...')
    autoencoder.train()
    for ae_epoch in range(50):
        total_loss = 0.0
        for (batch,) in ae_loader:
            batch = batch.to(device)
            ae_optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = ae_criterion(reconstructed, batch)
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()
        if (ae_epoch + 1) % 10 == 0:
            log(f'    AE epoch {ae_epoch+1}/50, loss: {total_loss/len(ae_loader):.4f}')

    # Extract features from eval samples
    autoencoder.eval()
    real_tensor = torch.tensor(real_g, dtype=torch.float32)
    with torch.no_grad():
        real_features = autoencoder.encode(real_tensor.to(device)).cpu().numpy()
        fake_tensor = torch.tensor(fake_g, dtype=torch.float32).to(device)
        fake_features = autoencoder.encode(fake_tensor).cpu().numpy()

    # Compute FID on autoencoder features
    mu_real, mu_fake = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False) + np.eye(64) * 1e-6
    cov_fake = np.cov(fake_features, rowvar=False) + np.eye(64) * 1e-6
    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(cov_real @ cov_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff**2) + np.trace(cov_real + cov_fake - 2*covmean)
    log(f'  FID score: {fid:.4f}')

    # Precision/Recall using Improved Precision and Recall Metric (Kynkäänniemi et al., 2019)
    # Vectorized implementation using cdist
    log(f'[9/9] Computing precision/recall (k=3)...')
    k = 3

    # Compute pairwise distances using cdist (vectorized)
    real_dists = cdist(real_flat_xy, real_flat_xy, metric='euclidean')
    fake_dists = cdist(fake_flat_xy, fake_flat_xy, metric='euclidean')
    real_fake_dists = cdist(real_flat_xy, fake_flat_xy, metric='euclidean')

    # k-NN radii: k-th smallest distance (excluding self which is 0)
    real_radii = np.sort(real_dists, axis=1)[:, k]  # k-th neighbor (index k since self is 0)
    fake_radii = np.sort(fake_dists, axis=1)[:, k]

    # Precision: fraction of fake samples within real manifold
    # For each fake sample, check if distance to any real sample <= that real's radius
    prec = np.mean([np.any(real_fake_dists[:, j] <= real_radii) for j in range(n)])
    # Recall: fraction of real samples within fake manifold
    rec = np.mean([np.any(real_fake_dists[i, :] <= fake_radii) for i in range(n)])
    log(f'  Precision: {prec:.3f}, Recall: {rec:.3f}')

    # Log to wandb
    import wandb
    os.environ['WANDB_API_KEY'] = WANDB_KEY
    wandb.init(project='wordgesture-gan', name=f'eval-epoch{epoch}', reinit=True)
    wandb.log({
        'eval/l2_wasserstein': l2_xy,
        'eval/dtw_wasserstein': dtw_xy,
        'eval/jerk': jerk_fake,
        'eval/velocity_corr': vcorr,
        'eval/accel_corr': acorr,
        'eval/duration_rmse_ms': duration_rmse_ms,
        'eval/fid': fid,
        'eval/precision': prec,
        'eval/recall': rec,
        'epoch': epoch
    })
    wandb.finish()

    log('\n' + '='*65)
    log(f'{"RESULTS (epoch "+str(epoch)+")":<30} {"Paper":<12} {"Ours":<12}')
    log('='*65)
    log(f'{"L2 Wasserstein (x,y)":<30} {4.409:<12.3f} {l2_xy:<12.3f}')
    log(f'{"DTW Wasserstein (x,y)":<30} {2.146:<12.3f} {dtw_xy:<12.3f}')
    log(f'{"Jerk (generated)":<30} {0.0058:<12.4f} {jerk_fake:<12.4f}')
    log(f'{"Velocity Correlation":<30} {0.40:<12.2f} {vcorr:<12.2f}')
    log(f'{"Acceleration Correlation":<30} {0.26:<12.2f} {acorr:<12.2f}')
    # Paper reports 1180.3ms. Show both sec and ms for comparison.
    log(f'{"Duration RMSE (ms)":<30} {1180.3:<12.1f} {duration_rmse_ms:<12.1f}')
    log(f'{"FID Score":<30} {0.270:<12.3f} {fid:<12.3f}')
    log(f'{"Precision":<30} {0.973:<12.3f} {prec:<12.3f}')
    log(f'{"Recall":<30} {0.258:<12.3f} {rec:<12.3f}')
    log('='*65)

    return {
        'epoch': int(epoch), 'l2_xy': float(l2_xy), 'dtw_xy': float(dtw_xy),
        'jerk_fake': float(jerk_fake), 'vcorr': float(vcorr), 'acorr': float(acorr),
        'duration_rmse_ms': float(duration_rmse_ms), 'fid': float(fid),
        'precision': float(prec), 'recall': float(rec)
    }


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=7200)
def evaluate_shark2_wer(n_train_user: int = 200, n_simulated: int = 0, n_test: int = 5000,
                        checkpoint_epoch: int = None, truncation: float = 0.05):
    """Evaluate SHARK2 decoder Word Error Rate (Section 5.10, Table 7 from paper).

    The SHARK2 decoder is a multi-channel recognition system that integrates:
    - Location channel: distance from gesture to word template (key positions)
    - Shape channel: shape similarity after normalizing for position/scale
    - Language model: unigram word probabilities

    Paper results (Table 7):
    - 200 User-drawn gestures: 32.8% WER
    - 200 User-drawn + 10000 simulated: 28.6% WER
    - 10000 Simulated only: 28.6% WER
    - 10000 User-drawn: 27.8% WER

    Args:
        n_train_user: Number of user-drawn gestures for training decoder parameters
        n_simulated: Number of simulated gestures to add to training set
        n_test: Number of test gestures for evaluation (default 5000 for speed)
        checkpoint_epoch: Specific checkpoint epoch for generator (default: latest)
        truncation: Latent truncation for generator
    """
    def log(msg):
        print(msg, flush=True)

    import torch
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    import random
    from joblib import Parallel, delayed

    from src.config import ModelConfig, TrainingConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, normalize_gesture
    from src.models import Generator

    device = 'cuda'
    log(f'[SHARK2] GPU: {torch.cuda.get_device_name(0)}')
    log(f'[SHARK2] Training setup: {n_train_user} user + {n_simulated} simulated gestures')
    log(f'[SHARK2] Test set: {n_test} gestures')

    # Load data
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    training_config = TrainingConfig()
    keyboard = QWERTYKeyboard()

    log('[1/6] Loading gesture dataset...')
    gestures_by_word, prototypes_by_word = load_dataset_from_zip(
        '/data/swipelogs.zip', keyboard, model_config, training_config
    )

    # Get all words with gestures
    all_words = list(gestures_by_word.keys())
    log(f'  Found {len(all_words)} words, {sum(len(g) for g in gestures_by_word.values())} total gestures')

    # Create vocabulary (lexicon) for decoder
    lexicon = list(all_words)  # Keep as list for indexing
    word_to_idx = {w: i for i, w in enumerate(lexicon)}
    n_words = len(lexicon)

    # Build unigram language model (uniform for now, or load frequencies)
    # Paper used 30k unigram from COCA corpus - we'll approximate with uniform
    log('[2/6] Building language model...')
    word_log_prob = np.full(n_words, np.log(1.0 / n_words))  # Uniform prior

    # Precompute word prototypes (templates) as numpy arrays
    log('[3/6] Precomputing word templates...')
    # Stack all templates: (n_words, seq_len, 2) for xy only
    all_templates_xy = np.stack([keyboard.get_word_prototype(w, model_config.seq_length)[:, :2]
                                  for w in lexicon], axis=0)

    # Precompute normalized shapes for all templates
    def normalize_shape_batch(templates):
        """Normalize shapes for batch of templates: (n, seq_len, 2) -> (n, seq_len, 2)."""
        # Center at origin
        centered = templates - templates.mean(axis=1, keepdims=True)
        # Path length for each
        diffs = np.diff(centered, axis=1)
        path_lengths = np.sum(np.sqrt(np.sum(diffs**2, axis=2)), axis=1, keepdims=True)
        path_lengths = np.maximum(path_lengths, 1e-6)  # Avoid division by zero
        # Scale
        return centered / path_lengths[:, :, np.newaxis]

    all_templates_shape = normalize_shape_batch(all_templates_xy)
    log(f'  Precomputed {n_words} word templates')

    # Vectorized SHARK2 decoder
    def decode_gestures_batch(gestures_xy, sigma_loc, sigma_shape, sigma_lm):
        """Decode multiple gestures using vectorized operations.

        Args:
            gestures_xy: (n_gestures, seq_len, 2) array of gesture x,y coordinates
            sigma_loc, sigma_shape, sigma_lm: SHARK2 parameters

        Returns:
            List of predicted word indices
        """
        n_gestures = len(gestures_xy)

        # Normalize gesture shapes
        gestures_shape = normalize_shape_batch(gestures_xy)

        predictions = []
        for i in range(n_gestures):
            g_xy = gestures_xy[i]  # (seq_len, 2)
            g_shape = gestures_shape[i]  # (seq_len, 2)

            # Location channel: L2 distance to all templates
            # (n_words, seq_len, 2) - (seq_len, 2) -> (n_words,)
            loc_dists = np.sqrt(np.mean((all_templates_xy - g_xy)**2, axis=(1, 2)))
            loc_scores = -loc_dists**2 / (2 * sigma_loc**2)

            # Shape channel: L2 distance to all normalized templates
            shape_dists = np.sqrt(np.mean((all_templates_shape - g_shape)**2, axis=(1, 2)))
            shape_scores = -shape_dists**2 / (2 * sigma_shape**2)

            # Total score
            scores = loc_scores + shape_scores + sigma_lm * word_log_prob

            # Best word
            best_idx = np.argmax(scores)
            predictions.append(best_idx)

        return predictions

    # Prepare train and test sets
    log('[4/6] Preparing train/test split...')

    # Flatten all gestures with their words
    all_gesture_word_pairs = []
    for word, gesture_list in gestures_by_word.items():
        for gesture in gesture_list:
            all_gesture_word_pairs.append((gesture, word))

    random.seed(42)
    random.shuffle(all_gesture_word_pairs)

    # Reserve test set first (cap at available gestures)
    n_test = min(n_test, len(all_gesture_word_pairs) - n_train_user - 100)
    test_pairs = all_gesture_word_pairs[:n_test]
    remaining_pairs = all_gesture_word_pairs[n_test:]

    # Sample training user gestures
    train_user_pairs = remaining_pairs[:n_train_user]

    log(f'  Train user gestures: {len(train_user_pairs)}')
    log(f'  Test gestures: {len(test_pairs)}')

    # Generate simulated gestures if needed
    train_simulated_pairs = []
    if n_simulated > 0:
        log(f'[4.5/6] Generating {n_simulated} simulated gestures...')

        # Load generator
        if checkpoint_epoch is not None:
            checkpoint_path = Path(f'/data/checkpoints/epoch_{checkpoint_epoch}.pt')
        else:
            checkpoint_path = Path('/data/checkpoints/latest.pt')

        if not checkpoint_path.exists():
            log(f'  ERROR: No checkpoint found at {checkpoint_path}')
            return {'error': 'No checkpoint found'}

        generator = Generator(model_config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        generator.eval()
        log(f'  Loaded generator from epoch {ckpt["epoch"] + 1}')

        # Generate gestures in batches for speed
        batch_size = 64
        words_list = lexicon
        with torch.no_grad():
            for batch_start in range(0, n_simulated, batch_size):
                batch_end = min(batch_start + batch_size, n_simulated)
                batch_words = [random.choice(words_list) for _ in range(batch_end - batch_start)]

                # Batch generate
                protos = torch.stack([torch.FloatTensor(
                    keyboard.get_word_prototype(w, model_config.seq_length)
                ) for w in batch_words]).to(device)
                z = torch.randn(len(batch_words), model_config.latent_dim, device=device) * truncation
                fakes = generator(protos, z).cpu().numpy()

                for j, word in enumerate(batch_words):
                    train_simulated_pairs.append((fakes[j], word))

                if (batch_end) % 2000 == 0:
                    log(f'    Generated {batch_end}/{n_simulated}')

        log(f'  Generated {len(train_simulated_pairs)} simulated gestures')

    # Combine training data
    train_pairs = train_user_pairs + train_simulated_pairs
    log(f'  Total training gestures: {len(train_pairs)}')

    # Train SHARK2 parameters using grid search
    log('[5/6] Training SHARK2 parameters...')

    def compute_wer_fast(pairs, sigma_loc, sigma_shape, sigma_lm):
        """Compute word error rate using vectorized decoder."""
        gestures_xy = np.array([g[:, :2] for g, _ in pairs])
        true_words = [w for _, w in pairs]
        true_indices = [word_to_idx[w] for w in true_words]

        pred_indices = decode_gestures_batch(gestures_xy, sigma_loc, sigma_shape, sigma_lm)
        errors = sum(1 for p, t in zip(pred_indices, true_indices) if p != t)
        return errors / len(pairs)

    # Grid search for optimal parameters
    # Use subset of training data for faster parameter search
    train_subset = train_pairs[:min(200, len(train_pairs))]

    best_params = None
    best_wer = 1.0

    # Parameter ranges based on typical gesture distances
    sigma_loc_range = [0.1, 0.2, 0.3, 0.5]
    sigma_shape_range = [0.05, 0.1, 0.2, 0.3]
    sigma_lm_range = [0.0, 0.1, 0.5, 1.0]

    log('  Grid search over parameters...')
    for sigma_loc in sigma_loc_range:
        for sigma_shape in sigma_shape_range:
            for sigma_lm in sigma_lm_range:
                wer = compute_wer_fast(train_subset, sigma_loc, sigma_shape, sigma_lm)
                if wer < best_wer:
                    best_wer = wer
                    best_params = (sigma_loc, sigma_shape, sigma_lm)
                    log(f'    New best: loc={sigma_loc}, shape={sigma_shape}, lm={sigma_lm}, WER={wer*100:.1f}%')

    sigma_loc, sigma_shape, sigma_lm = best_params
    log(f'  Best params: sigma_loc={sigma_loc}, sigma_shape={sigma_shape}, sigma_lm={sigma_lm}')
    log(f'  Train WER (subset): {best_wer * 100:.1f}%')

    # Evaluate on test set in batches
    log(f'[6/6] Evaluating on {len(test_pairs)} test gestures...')

    test_gestures_xy = np.array([g[:, :2] for g, _ in test_pairs])
    test_true_indices = [word_to_idx[w] for _, w in test_pairs]

    # Process in batches for progress reporting
    batch_size = 500
    test_errors = 0
    for batch_start in range(0, len(test_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(test_pairs))
        batch_gestures = test_gestures_xy[batch_start:batch_end]
        batch_true = test_true_indices[batch_start:batch_end]

        batch_pred = decode_gestures_batch(batch_gestures, sigma_loc, sigma_shape, sigma_lm)
        batch_errors = sum(1 for p, t in zip(batch_pred, batch_true) if p != t)
        test_errors += batch_errors

        log(f'  Evaluated {batch_end}/{len(test_pairs)}, current WER: {test_errors / batch_end * 100:.1f}%')

    test_wer = test_errors / len(test_pairs)

    log('\n' + '='*65)
    log(f'SHARK2 DECODER RESULTS')
    log('='*65)
    log(f'Training Setup: {n_train_user} user + {n_simulated} simulated gestures')
    log(f'Test Set: {len(test_pairs)} gestures')
    log(f'Parameters: sigma_loc={sigma_loc}, sigma_shape={sigma_shape}, sigma_lm={sigma_lm}')
    log(f'Word Error Rate: {test_wer * 100:.1f}%')
    log('='*65)
    log('\nPaper Reference (Table 7):')
    log(f'  200 User-drawn: 32.8% WER')
    log(f'  200 User + 10000 Simulated: 28.6% WER')
    log(f'  10000 Simulated only: 28.6% WER')
    log(f'  10000 User-drawn: 27.8% WER')
    log('='*65)

    # Log to wandb
    import wandb
    os.environ['WANDB_API_KEY'] = WANDB_KEY
    wandb.init(project='wordgesture-gan', name=f'shark2-{n_train_user}u-{n_simulated}s', reinit=True)
    wandb.log({
        'shark2/wer': test_wer,
        'shark2/n_train_user': n_train_user,
        'shark2/n_simulated': n_simulated,
        'shark2/n_test': len(test_pairs),
        'shark2/sigma_loc': sigma_loc,
        'shark2/sigma_shape': sigma_shape,
        'shark2/sigma_lm': sigma_lm,
    })
    wandb.finish()

    return {
        'wer': float(test_wer),
        'n_train_user': n_train_user,
        'n_simulated': n_simulated,
        'n_test': len(test_pairs),
        'sigma_loc': float(sigma_loc),
        'sigma_shape': float(sigma_shape),
        'sigma_lm': float(sigma_lm),
    }


async def main():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--checkpoint-epoch', type=int, default=None, help='Specific epoch checkpoint to evaluate')
    parser.add_argument('--shark2', action='store_true', help='Run SHARK2 decoder WER evaluation')
    parser.add_argument('--shark2-train-user', type=int, default=200, help='Number of user gestures for SHARK2 training')
    parser.add_argument('--shark2-simulated', type=int, default=0, help='Number of simulated gestures for SHARK2 training')
    args = parser.parse_args()

    # Enable streaming logs from Modal containers
    modal.enable_output()
    print('[local] Starting Modal app...')
    start_time = time.time()

    async with app.run():
        print(f'[local] Modal app running ({time.time() - start_time:.1f}s), dispatching task...')
        if args.shark2:
            print(f'[local] Calling evaluate_shark2_wer.remote.aio()...')
            result = await evaluate_shark2_wer.remote.aio(
                n_train_user=args.shark2_train_user,
                n_simulated=args.shark2_simulated,
                checkpoint_epoch=args.checkpoint_epoch
            )
            print(f'[local] SHARK2 evaluation completed ({time.time() - start_time:.1f}s)')
        elif args.eval_only:
            print(f'[local] Calling evaluate.remote.aio()...')
            result = await evaluate.remote.aio(checkpoint_epoch=args.checkpoint_epoch)
            print(f'[local] evaluate completed ({time.time() - start_time:.1f}s)')
        else:
            print(f'[local] Calling train.remote.aio()...')
            result = await train.remote.aio(num_epochs=args.epochs, resume=not args.no_resume)
            print(f'[local] train completed ({time.time() - start_time:.1f}s)')
            if result.get('status') == 'complete':
                print('\nRunning evaluation...')
                result['eval'] = await evaluate.remote.aio()
    print(f'\n[local] Total time: {time.time() - start_time:.1f}s')
    print(f'Result: {result}')


if __name__ == '__main__':
    asyncio.run(main())
