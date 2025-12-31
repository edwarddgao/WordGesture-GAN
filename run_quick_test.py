#!/usr/bin/env python3
"""Quick 10-epoch test to verify training + eval pipeline works."""

import os
if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch
import modal
import asyncio

app = modal.App('wordgesture-quick-test')
volume = modal.Volume.from_name('wordgesture-data')

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch>=2.0.0', 'numpy>=1.24.0', 'scipy>=1.10.0')
)


@app.function(gpu='T4', image=image, volumes={'/data': volume}, timeout=600)
def quick_train_eval():
    """Quick 10-epoch run to test pipeline."""
    import sys
    sys.path.insert(0, '/data')

    import torch
    import numpy as np
    import random
    from scipy.optimize import linear_sum_assignment

    from src.config import ModelConfig, TrainingConfig
    from src.keyboard import QWERTYKeyboard
    from src.data import load_dataset_from_zip, create_train_test_split, create_data_loaders
    from src.trainer import WordGestureGANTrainer

    device = 'cuda'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Config - 10 epochs only
    model_config = ModelConfig(seq_length=128, latent_dim=32)
    training_config = TrainingConfig(batch_size=512, num_epochs=10, n_critic=5)

    # Data
    keyboard = QWERTYKeyboard()
    gestures, protos = load_dataset_from_zip('/data/swipelogs.zip', keyboard, model_config, training_config)
    train_ds, test_ds = create_train_test_split(gestures, protos, train_ratio=0.8, seed=seed)
    train_loader, _ = create_data_loaders(train_ds, test_ds, batch_size=512, num_workers=2)
    print(f'Data: {len(train_ds)} train, {len(test_ds)} test')

    # Train
    trainer = WordGestureGANTrainer(model_config, training_config, device)
    print('Training 10 epochs...')

    for epoch in range(10):
        trainer.generator.train()
        trainer.encoder.train()
        d1_sum, rec_sum, n = 0, 0, 0
        for batch in train_loader:
            real = batch['gesture'].to(device)
            proto = batch['prototype'].to(device)
            bs = real.size(0)

            # D1
            trainer.optimizer_D1.zero_grad()
            z = torch.randn(bs, 32, device=device)
            with torch.no_grad():
                fake = trainer.generator(proto, z)
            d1_loss = -trainer.discriminator_1(real).mean() + trainer.discriminator_1(fake).mean()
            d1_loss.backward()
            trainer.optimizer_D1.step()

            # G+E
            trainer.optimizer_G.zero_grad()
            trainer.optimizer_E.zero_grad()
            z = torch.randn(bs, 32, device=device)
            fake = trainer.generator(proto, z)
            g_loss = -trainer.discriminator_1(fake).mean()
            z_enc, mu, lv = trainer.encoder(real)
            fake2 = trainer.generator(proto, z_enc)
            rec_loss = ((real - fake2) ** 2).mean()
            (g_loss + 5.0 * rec_loss).backward()
            trainer.optimizer_G.step()
            trainer.optimizer_E.step()

            d1_sum += d1_loss.item()
            rec_sum += rec_loss.item()
            n += 1
        print(f'Epoch {epoch+1}/10 - D1:{d1_sum/n:.2f} rec:{rec_sum/n:.4f}')

    # Quick eval - just L2 distance on 100 samples
    print('Quick eval...')
    trainer.generator.eval()
    with torch.no_grad():
        # Get 100 test samples
        test_real, test_fake = [], []
        for i in range(min(100, len(test_ds))):
            item = test_ds[i]
            proto = item['prototype'].unsqueeze(0).to(device)
            z = torch.randn(1, 32, device=device)
            fake = trainer.generator(proto, z).cpu().numpy()[0]
            test_real.append(item['gesture'].numpy())
            test_fake.append(fake)

    test_real = np.array(test_real)
    test_fake = np.array(test_fake)

    # L2 distance
    n = len(test_real)
    d = np.array([[np.sqrt(np.sum((test_real[i, :, :2] - test_fake[j, :, :2]) ** 2))
                   for j in range(n)] for i in range(n)])
    r, c = linear_sum_assignment(d)
    l2_xy = d[r, c].mean()

    print(f'L2 Wasserstein (x,y): {l2_xy:.3f} (paper: 4.409)')
    return {'l2_xy': float(l2_xy), 'epochs': 10}


async def main():
    print('Quick test: 10 epochs + eval')
    async with app.run():
        result = await quick_train_eval.remote.aio()
    print(f'Result: {result}')
    return result


if __name__ == '__main__':
    asyncio.run(main())
