import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def get_training_strategy(model, max_lr, warmup_steps, num_steps):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,   # initial lr = 0.1 * base_lr
                end_factor=1.0,
                total_iters=warmup_steps
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=num_steps - warmup_steps,
                eta_min=0.1 * max_lr
            )
        ],
        milestones=[warmup_steps]
    )

    return loss_fn, optimizer, scheduler