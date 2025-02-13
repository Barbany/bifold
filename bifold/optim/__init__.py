import torch


class Optimizers:
    @staticmethod
    def get_by_name(cfg, params):
        if cfg.name == "adam":
            del cfg.name
            return torch.optim.Adam(params, **cfg)
        if cfg.name == "adamw":
            del cfg.name
            return torch.optim.AdamW(params, **cfg)
        else:
            raise ValueError(f"Optimizer {cfg.name} not recognized")


class Schedulers:
    @staticmethod
    def get_by_name(cfg, *args, **kwargs):
        if cfg.name is None:
            return None
        elif cfg.name == "linear_warmup":
            return LinearWarmup(cfg, *args, **kwargs)
        else:
            raise ValueError(f"Scheduler {cfg.name} not recognized")


class LinearWarmup:
    def __init__(self, cfg, optimizer, max_iters):
        self.warmup_steps = int(cfg.warmup_portion * max_iters)
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = -1
        if isinstance(cfg.warmup_start_lr, (list, tuple)):
            assert len(cfg.warmup_start_lr) == len(self.base_lrs), (
                f"The length of warmup_start_lr {len(cfg.warmup_start_lr)} "
                f"and optimizer.param_group {len(self.base_lrs)} do not correspond"
            )
            self.warmup_start_lrs = cfg.warmup_start_lr

        else:
            self.warmup_start_lrs = [cfg.warmup_start_lr] * len(self.base_lrs)

        if cfg.use_cosine_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max_iters - self.warmup_steps
            )
        else:
            self.lr_scheduler = None

        self.step()

    def get_last_lr(self):
        r"""
        Return last computed learning rate by current warmup_scheduler_pytorch scheduler.
        """
        return self._last_lr

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_scheduler")
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        if self.last_step <= self.warmup_steps:
            values = [
                warmup_lr + (base_lr - warmup_lr) * (self.last_step / self.warmup_steps)
                for warmup_lr, base_lr in zip(self.warmup_start_lrs, self.base_lrs)
            ]
            for idx, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
                param_group["lr"] = lr
            self.last_step += 1
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
