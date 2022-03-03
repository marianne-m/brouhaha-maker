import math
from statistics import median


class BalanceSampler:
    def __call__(self, category_sizes):
        raise RuntimeError("Not Implemented")

    def readjust_dsitrib(self, old_sizes, new_sizes):
        tot_size_in = sum(old_sizes)
        tot_size_out = sum(new_sizes)
        factor = tot_size_in / tot_size_out
        return [int(factor * x) for x in new_sizes]


class LinearBalance(BalanceSampler):
    def __init__(self, balance_coeff):
        self.balance_coeff = balance_coeff

    def __call__(self, category_sizes):
        target_val = median(category_sizes)
        return [
            int(self.balance_coeff * target_val + (1 - self.balance_coeff) * x)
            for x in category_sizes
        ]


class LogBalance(BalanceSampler):
    def __call__(self, category_sizes):
        tmp_out = [math.log(x + 1) for x in category_sizes]
        return self.readjust_dsitrib(category_sizes, tmp_out)


class PowBalance(BalanceSampler):
    def __init__(self, pow_val=0.5):
        self.pow_val = pow_val

    def __call__(self, category_sizes):
        if self.pow_val == 0.5:
            tmp_out = [int(math.sqrt(x)) for x in category_sizes]
        else:
            tmp_out = [int(math.pow(x, self.pow_val)) for x in category_sizes]
        return self.readjust_dsitrib(category_sizes, tmp_out)


def get_balance_sampler(sampler_name, **kwargs):

    if sampler_name == "linear":
        return LinearBalance(kwargs["balance_coeff"])
    if sampler_name == "log":
        return LogBalance()
    if sampler_name == "pow":
        return PowBalance(kwargs["balance_coeff"])
