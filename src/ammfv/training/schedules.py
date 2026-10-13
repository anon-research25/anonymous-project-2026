import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(optimizer, total_steps, warmup_ratio=0.1):
    warmup = int(total_steps * warmup_ratio)

    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def annealed_temperature(epoch, initial, minimum, decay):
    return max(minimum, initial * decay**epoch)
