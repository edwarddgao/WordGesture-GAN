#!/usr/bin/env python3
"""Train WordGesture-GAN with wandb logging and evaluation."""

import modal_proxy_patch  # Must be first
import modal
import asyncio
from datetime import datetime

app = modal.App('wordgesture-gan-experiment')
volume = modal.Volume.from_name('wordgesture-data')

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0',
                 'matplotlib>=3.7.0', 'wandb', 'pillow')
)

WANDB_KEY = 'd68f9b4406a518b2095a579d37b0355bc18ad1a8'


@app.function(
    gpu='T4',
    image=image,
    volumes={'/data': volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({'WANDB_API_KEY': WANDB_KEY})]
)
def train_and_evaluate(num_epochs: int = 200):
    import sys
    sys.path.insert(0, '/data')

    import torch
    import numpy as np
    import random
    import wandb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    from datetime import datetime
    from scipy.optimize import linear_sum_assignment

    from src.config import ModelConfig, TrainingConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
    from src.trainer import WordGestureGANTrainer

    # Setup
    device = 'cuda'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Wandb init
    run = wandb.init(
        project='wordgesture-gan',
        name=f'wg-gan-{num_epochs}ep-{datetime.now().strftime("%m%d-%H%M")}',
        config={
            'epochs': num_epochs,
            'batch_size': 512,
            'learning_rate': 0.0002,
            'latent_dim': 32,
            'n_critic': 5,
            'lambda_feat': 1.0,
            'lambda_rec': 5.0,
            'lambda_lat': 0.5,
            'lambda_kld': 0.05,
        }
    )
    print(f'Wandb: {run.url}')

    # Config
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    training_config = TrainingConfig(
        batch_size=512,
        learning_rate=0.0002,
        num_epochs=num_epochs,
        n_critic=5,
        lambda_feat=1.0,
        lambda_rec=5.0,
        lambda_lat=0.5,
        lambda_kld=0.05
    )

    # Data
    keyboard = QWERTYKeyboard()
    gestures, protos = load_dataset_from_zip(
        '/data/swipelogs.zip', keyboard, model_config, training_config
    )
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=seed)
    train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=512, num_workers=2)
    print(f'Data: {len(train_ds)} train, {len(test_ds)} test')

    # Model
    trainer = WordGestureGANTrainer(model_config, training_config, device)

    # Visualization words
    viz_words = ['hello', 'world', 'the', 'place', 'quick']
    viz_protos = torch.FloatTensor(
        np.stack([keyboard.get_word_prototype(w, 128) for w in viz_words])
    ).to(device)

    def log_gesture_images(epoch):
        """Generate and log gesture visualizations to wandb."""
        trainer.generator.eval()
        with torch.no_grad():
            z = torch.randn(len(viz_words), 32, device=device)
            fakes = trainer.generator(viz_protos, z).cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, (word, g) in enumerate(zip(viz_words, fakes)):
            # Plot gesture path
            axes[i].plot(g[:, 0], -g[:, 1], 'b-', lw=2, alpha=0.8)
            # Color points by time (velocity visualization)
            colors = np.linspace(0, 1, len(g[::8]))
            axes[i].scatter(g[::8, 0], -g[::8, 1], c=colors, cmap='plasma', s=40, zorder=5)
            axes[i].set_title(f'"{word}"', fontsize=14, fontweight='bold')
            axes[i].set_xlim(-1.3, 1.3)
            axes[i].set_ylim(-1.3, 1.3)
            axes[i].set_aspect('equal')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.suptitle(f'Generated Gestures - Epoch {epoch}', fontsize=16)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()

        return wandb.Image(Image.open(buf), caption=f'Epoch {epoch}')

    # Training loop
    print(f'\nStarting {num_epochs} epoch training...')
    t0 = datetime.now()

    for epoch in range(num_epochs):
        trainer.generator.train()
        trainer.encoder.train()
        trainer.discriminator_1.train()
        trainer.discriminator_2.train()

        d1_sum, d2_sum, rec_sum, lat_sum, kld_sum = 0, 0, 0, 0, 0
        n_batches = 0

        for batch in train_loader:
            real = batch['gesture'].to(device)
            proto = batch['prototype'].to(device)
            bs = real.size(0)

            # Train discriminators (n_critic times)
            for _ in range(training_config.n_critic):
                # D1: random z
                trainer.optimizer_D1.zero_grad()
                z = torch.randn(bs, 32, device=device)
                with torch.no_grad():
                    fake1 = trainer.generator(proto, z)
                d1_loss = -trainer.discriminator_1(real).mean() + trainer.discriminator_1(fake1).mean()
                d1_loss.backward()
                trainer.optimizer_D1.step()

                # D2: encoded z
                trainer.optimizer_D2.zero_grad()
                with torch.no_grad():
                    z_enc, _, _ = trainer.encoder(real)
                    fake2 = trainer.generator(proto, z_enc)
                d2_loss = -trainer.discriminator_2(real).mean() + trainer.discriminator_2(fake2).mean()
                d2_loss.backward()
                trainer.optimizer_D2.step()

            # Train generator and encoder
            trainer.optimizer_G.zero_grad()
            trainer.optimizer_E.zero_grad()

            # Cycle 1: z -> X' -> z'
            z = torch.randn(bs, 32, device=device)
            fake1 = trainer.generator(proto, z)
            g1_loss = -trainer.discriminator_1(fake1).mean()
            z_rec, _, _ = trainer.encoder(fake1)
            lat_loss = ((z - z_rec) ** 2).mean()

            # Cycle 2: X -> z -> X'
            z_enc, mu, lv = trainer.encoder(real)
            fake2 = trainer.generator(proto, z_enc)
            g2_loss = -trainer.discriminator_2(fake2).mean()
            rec_loss = ((real - fake2) ** 2).mean()
            kld_loss = -0.5 * (1 + lv - mu ** 2 - lv.exp()).mean()

            # Total loss
            total_loss = (g1_loss + g2_loss +
                         training_config.lambda_lat * lat_loss +
                         training_config.lambda_rec * rec_loss +
                         training_config.lambda_kld * kld_loss)
            total_loss.backward()
            trainer.optimizer_G.step()
            trainer.optimizer_E.step()

            d1_sum += d1_loss.item()
            d2_sum += d2_loss.item()
            rec_sum += rec_loss.item()
            lat_sum += lat_loss.item()
            kld_sum += kld_loss.item()
            n_batches += 1

        # Log metrics every epoch
        wandb.log({
            'epoch': epoch + 1,
            'loss/d1': d1_sum / n_batches,
            'loss/d2': d2_sum / n_batches,
            'loss/reconstruction': rec_sum / n_batches,
            'loss/latent': lat_sum / n_batches,
            'loss/kld': kld_sum / n_batches,
        })

        # Log images every 5 epochs
        if (epoch + 1) % 5 == 0:
            wandb.log({'generated_gestures': log_gesture_images(epoch + 1)})
            elapsed = (datetime.now() - t0).total_seconds() / 60
            print(f'Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}m) - '
                  f'D1:{d1_sum/n_batches:.2f} D2:{d2_sum/n_batches:.2f} '
                  f'rec:{rec_sum/n_batches:.4f}')

    train_time = (datetime.now() - t0).total_seconds() / 60
    print(f'\nTraining complete: {train_time:.1f} min')

    # ============ EVALUATION ============
    print('\nRunning evaluation...')

    # Organize test data by word
    test_by_word = {}
    for i in range(len(test_ds)):
        item = test_ds[i]
        key = tuple(item['prototype'][0].numpy().round(3))
        if key not in test_by_word:
            test_by_word[key] = {'real': [], 'proto': item['prototype'].numpy()}
        test_by_word[key]['real'].append(item['gesture'].numpy())

    # Generate gestures for test set
    all_real, all_fake = [], []
    trainer.generator.eval()
    with torch.no_grad():
        for data in test_by_word.values():
            n = len(data['real'])
            proto = torch.FloatTensor(data['proto']).unsqueeze(0).repeat(n, 1, 1).to(device)
            z = torch.randn(n, 32, device=device)
            fake = trainer.generator(proto, z).cpu().numpy()
            all_real.extend(data['real'])
            all_fake.extend(fake)

    all_real = np.array(all_real)
    all_fake = np.array(all_fake)
    print(f'Generated {len(all_fake)} test gestures')

    # L2 Wasserstein Distance
    n = min(1500, len(all_real))
    idx = np.random.choice(len(all_real), n, replace=False)
    rs, fs = all_real[idx], all_fake[idx]

    # (x, y) only
    d_xy = np.array([[np.sqrt(np.sum((rs[i, :, :2] - fs[j, :, :2]) ** 2))
                      for j in range(n)] for i in range(n)])
    r, c = linear_sum_assignment(d_xy)
    l2_xy = d_xy[r, c].mean()
    l2_xy_std = d_xy[r, c].std()

    # (x, y, t)
    d_xyt = np.array([[np.sqrt(np.sum((rs[i] - fs[j]) ** 2))
                       for j in range(n)] for i in range(n)])
    r, c = linear_sum_assignment(d_xyt)
    l2_xyt = d_xyt[r, c].mean()

    # DTW Wasserstein
    def dtw(s1, s2):
        n, m = len(s1), len(s2)
        d = np.full((n + 1, m + 1), np.inf)
        d[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.sqrt(np.sum((s1[i-1] - s2[j-1]) ** 2))
                d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
        return d[n, m]

    nd = min(400, n)
    dtw_mat = np.array([[dtw(rs[i, :, :2], fs[j, :, :2]) for j in range(nd)] for i in range(nd)])
    r, c = linear_sum_assignment(dtw_mat)
    dtw_xy = dtw_mat[r, c].mean()

    # Jerk
    def compute_jerk(g):
        dt = np.diff(g[:, 2])
        dt[dt == 0] = 1e-6
        vx, vy = np.diff(g[:, 0]) / dt, np.diff(g[:, 1]) / dt
        ax, ay = np.diff(vx) / dt[1:], np.diff(vy) / dt[1:]
        jx, jy = np.diff(ax) / dt[2:], np.diff(ay) / dt[2:]
        return np.mean(np.sqrt(jx ** 2 + jy ** 2))

    jerk_real = np.nanmean([compute_jerk(g) for g in all_real[:500]])
    jerk_fake = np.nanmean([compute_jerk(g) for g in all_fake[:500]])

    # Velocity correlation
    def compute_velocity(g):
        dt = np.diff(g[:, 2])
        dt[dt == 0] = 1e-6
        return np.sqrt((np.diff(g[:, 0]) / dt) ** 2 + (np.diff(g[:, 1]) / dt) ** 2)

    vcorrs = [np.corrcoef(compute_velocity(all_real[i]), compute_velocity(all_fake[i]))[0, 1]
              for i in range(min(500, len(all_real)))]
    vcorr = np.nanmean(vcorrs)

    # Precision / Recall (k=3)
    npr = min(800, len(all_real))
    rf = all_real[:npr].reshape(npr, -1)
    ff = all_fake[:npr].reshape(npr, -1)

    prec = sum(1 for i in range(npr)
               if np.linalg.norm(rf - ff[i], axis=1).min() <=
               np.partition(np.linalg.norm(rf - ff[i], axis=1), 3)[3]) / npr
    rec = sum(1 for i in range(npr)
              if np.linalg.norm(ff - rf[i], axis=1).min() <=
              np.partition(np.linalg.norm(ff - rf[i], axis=1), 3)[3]) / npr

    # Log final metrics
    wandb.log({
        'eval/l2_wasserstein_xy': l2_xy,
        'eval/l2_wasserstein_xyt': l2_xyt,
        'eval/dtw_wasserstein_xy': dtw_xy,
        'eval/jerk_real': jerk_real,
        'eval/jerk_generated': jerk_fake,
        'eval/velocity_correlation': vcorr,
        'eval/precision': prec,
        'eval/recall': rec,
    })

    # Print comparison with paper
    print('\n' + '=' * 65)
    print(f'{"Metric":<28} {"Paper":<12} {"Ours":<12}')
    print('-' * 65)
    print(f'{"L2 Wasserstein (x,y)":<28} {4.409:<12.3f} {l2_xy:<12.3f}')
    print(f'{"L2 Wasserstein (x,y,t)":<28} {4.426:<12.3f} {l2_xyt:<12.3f}')
    print(f'{"DTW Wasserstein (x,y)":<28} {2.146:<12.3f} {dtw_xy:<12.3f}')
    print(f'{"Jerk (user-drawn)":<28} {0.0066:<12.4f} {jerk_real:<12.4f}')
    print(f'{"Jerk (generated)":<28} {0.0058:<12.4f} {jerk_fake:<12.4f}')
    print(f'{"Velocity Correlation":<28} {0.40:<12.2f} {vcorr:<12.2f}')
    print(f'{"Precision":<28} {0.973:<12.3f} {prec:<12.3f}')
    print(f'{"Recall":<28} {0.258:<12.3f} {rec:<12.3f}')
    print('=' * 65)

    wandb.finish()

    return {
        'l2_xy': l2_xy,
        'l2_xyt': l2_xyt,
        'dtw_xy': dtw_xy,
        'jerk_real': jerk_real,
        'jerk_fake': jerk_fake,
        'vcorr': vcorr,
        'precision': prec,
        'recall': rec,
        'train_time_min': train_time,
        'wandb_url': run.url
    }


async def main():
    print('Launching WordGesture-GAN experiment on Modal...')
    print('=' * 50)

    async with app.run():
        result = await train_and_evaluate.remote.aio(num_epochs=200)

    print(f'\nView results at: {result["wandb_url"]}')
    return result


if __name__ == '__main__':
    asyncio.run(main())
