"""
# MULTI-LOSS WEIGHTING WITH COEFFICIENT OF VARIATIONS
## https://arxiv.org/pdf/2009.01717.pdf

In other words, this hypothesis says that a loss with a constant value should not be optimised any further. Variance alone,
however, is not sufficient, given that it can be expected that a loss which has a larger (mean) magnitude, also has a higher
absolute variance. Even if the loss is relatively less variant. Therefore, we propose to use a dimensionless measure of
uncertainty, the Coefficient of Variation (cv, which shows the variability of the data in relation to the (observed) mean:
```
cv = σ/µ, (2)
```
where `µ` denotes the mean and `σ` the standard deviation. It allows to fairly compare the uncertainty in losses, even
with different magnitudes, under the assumption that each loss measures in a ratio-scale, that is with a unique and
non-arbitrary zero value.

Here a more robust loss ratio is proposed:
```
          Li(t)
li(t) = --------
        µLi(t − 1)
```
*(3)*
where `µLi(t − 1)` is the mean over all observed losses from iteration 1 to (`t` - 1) for a specific loss Li. The loss ratio l(t)
has the same meaningful zero point when `Li(t)` is zero and is a relative comparison of two measurements of the loss
statistic. Now, the loss ratio is used as a point estimate of the mean to yield the following definition for loss weights:
```
     σli(t)
αi = ------
      li(t)
```
*(4)*
where `σli(t)` is the standard deviation over all known loss ratios `Li(t)/µli(t−1)` until iteration `t` - 1
"""
import torch
import numpy as np

class LossTracker:
    def __init__(self, name, 
                 fn=lambda x, y: x,
                 experiment=None, 
                 weight=1.0,
                 warmup=np.inf,
                 loss_limits=[-np.inf, np.inf],
                 block_size=100,
                 scale_range=[0.2, 5],
                 use_ratio_scaling=False
                  ):
        """Wrapper around a call to a PyTorch Loss result that
        tracks the historical loss values and calculates statistics
        that can be used to automatically scale the applied loss
        value.  Logs of each update (including statistics) are sent to a comet.ml experiment
        if one is passed in.

        Args:
            name (str): Identifier for this particular loss function.
            fn (callable): Metric function to use as loss.
            experiment (comet.Experiment, optional): A comet.ml experiment for logging.
                Defaults to None.
            weight (float, optional): Static value to multiply each loss result by.
                Either this value or a dynamic weight will be applied, depending on the
                `warmup` value set.
                Defaults to 1.0.
            warmup (int, optional): Number of updates to wait before applying dynamic
                scaling value instead of the static `weight` scaling value.
                Defaults to np.inf.
            loss_limit (list, optional): The minimum and maximum values to restrict
                final loss values to. Defaults to [-np.inf, np.inf].
            block_size (int, optional): Number of elements to allocate to the numpy
                buffer used to store historical values. Defaults to 100.
            use_scaling (bool, optional): Apply dynamic scaling based on the ratio of current value to mean. 
                Defaults to False.
            scale_range (list, optional): Minimum and maximum values to restrict dynamic scaling by. 
                This prevents massive scaling values for irregular loss functions. 
                Defaults to [0.2, 5].
        """
        self.name = name
        self.fn = fn
        self.exp = experiment
        self.weight = weight
        self.loss_low_limit = loss_limits[0]
        self.loss_up_limit = loss_limits[1]
        self.warmup = warmup
        self.block_size = block_size
        self.use_scaling = use_ratio_scaling
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
        self.reset()

    def reset(self):
        self.last_value = None
        self.max = -np.inf
        self.min = np.inf
        self.mean = 1
        self.var = 0
        self.std = 0
        self.ratio = 0
        self.ratio_std = 0
        self.cov = 0
        self.cov_weight = self.weight
        self.value_history = np.empty(self.block_size)
        self.ratio_history = np.empty(self.block_size)
        self.max_history_size = self.block_size
        self.value = 0
        self.total = 0
        self.count = 0

    def expand_buffer(self, block_size=None):
        if block_size is not None:
            self.block_size = block_size

        self.value_history = np.concatenate((self.value_history, np.empty(self.block_size)))
        self.ratio_history = np.concatenate((self.ratio_history, np.empty(self.block_size)))
        self.max_history_size += self.block_size

    def __call__(self, x, y, 
               do_backwards=True,
               do_comet=True,
               do_console=False):
        self.last_value = self.fn(x, y)
        self.update(self.last_value, 
               do_backwards=do_backwards,
               do_comet=do_comet,
               do_console=do_console)
        return self.last_value
        
    def update(self, value, 
               do_backwards=True,
               do_comet=True,
               do_console=False):
        if self.use_scaling: value = self.adjust_loss(value, self.ratio)
        value = self.adjust_loss(value, self.weight)
        value = self.constrain_loss(value)
        if do_backwards:
            value.backward()
            self.value = value.item()
        else:
            self.value = value
        self.total += self.value
        if self.count >= self.max_history_size:
            self.expand_buffer()
        assert self.count < self.max_history_size
        self.value_history[self.count] = self.value

        # calculate li(t) - equation (3)
        if self.mean != 0:
            self.ratio = self.value / self.mean  # µLi(t − 1) is the mean over all observed losses from iteration 1 to (t - 1) for a specific loss Li
        else:
            self.ratio = 1 # ratio of 1 when mean is 0
        self.ratio = min(max(self.ratio, self.scale_min), self.scale_max)
        self.ratio_history[self.count] = self.ratio
        self.count += 1
        if self.count > 1:  # only once there is a history
            self.ratio_std = self.ratio_history[:self.count].std() # σli(t) is the standard deviation over all known loss ratios Li(t)/µli(t−1) until iteration t - 1
            self.cov_weight = self.ratio_std / self.ratio # (4):  αi = σli(t) / li(t)
        if self.count > self.warmup:
            # use cov weight as functioning weight after warmup period to allow for meaningful statistics to build
            self.weight = self.cov_weight
       
        # update comet or print out
        self.log(comet=do_comet, console=do_console)

    def set_stats(self):
        try:
            self.max = self.value_history[:self.count].max()
            self.min = self.value_history[:self.count].min()
            self.mean = self.value_history[:self.count].mean()
            self.var = self.value_history[:self.count].var()
            self.std = self.value_history[:self.count].std()
            self.cov = self.std / self.mean
        except Exception as e:
            print(f"[ERROR]\tFailed to set metric stats\n{e}")

    def log(self, comet=True, console=False):
        if comet and self.exp:
            self.exp.log_metric(f"{self.name}_loss", self.value)
            self.exp.log_metric(f"{self.name}_cov", self.cov)
            self.exp.log_metric(f"{self.name}_cov_weight", self.cov_weight)
            self.exp.log_metric(f"{self.name}_var", self.var)
            self.exp.log_metric(f"{self.name}_std", self.std)
        if console:
            msg = f"[{self.name}] [{self.count}]\t{self.value} @ {self.weight}x \t ~ mean: {self.mean} var: {self.var} std: {self.std} cov: {self.cov}"
            print(msg)
            if self.exp: 
                self.exp.log_text(msg)

    def get_history(self):
        return self.value_history[:self.count]

    def adjust_loss(self, loss, amount):
        """Changes the `loss` value by multiplying
        by `amount`.

        Args:
            loss (torch.Tensor | float): The loss result to scale
            amount (float): The magnitude to scale the loss by

        Returns:
            torch.Tensor | float: Scaled value
        """
        loss *= amount
        return loss

    def constrain_loss(self, loss):
        if not isinstance(loss, torch.Tensor):
            loss = torch.Tensor(loss)
        # soft restriction on loss value
        # loss is too big
        if loss > self.loss_up_limit:
            # magnitude is greater than 1 since loss is greater than upper limit
            magnitude = torch.floor(loss / self.loss_up_limit)
            # loss decreases since magnitude is greater than 1
            loss = loss / max(magnitude, 10) # sanity check, restrict decrease to 10x
        # loss is too small
        if loss < self.loss_low_limit:
            # magnitude is less than 1 since loss is less than lower limit
            magnitude = torch.floor(loss / self.loss_low_limit)
            # loss increases since magnitude is less than 1
            loss = loss * abs(magnitude) # sanity check, prevent sign flipping
        # hard restriction on loss value
        loss = torch.clamp(loss, self.loss_low_limit, self.loss_up_limit)
        return loss