import sys
sys.path.append('/workspace/comate_paddle_demo/代码转换_3/paddle_project/utils')
import paddle_aux
import paddle
import math


class AdaBound(paddle.optimizer.Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), final_lr=0.1,
        gamma=0.001, eps=1e-08, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format
                (betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format
                (betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError('Invalid final learning rate: {}'.format(final_lr)
                )
        if not 0.0 <= gamma < 1.0:
            raise ValueError('Invalid gamma parameter: {}'.format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma,
            eps=eps, weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse():
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                        )
                amsbound = group['amsbound']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = paddle.zeros_like(x=p.data)
                    state['exp_avg_sq'] = paddle.zeros_like(x=p.data)
                    if amsbound:
                        state['max_exp_avg_sq'] = paddle.zeros_like(x=p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.multiply_(y=paddle.to_tensor(beta1)).add_(1 - beta1,
                    grad)
                exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).addcmul_(1 -
                    beta2, grad, grad)
                if amsbound:
                    paddle_aux.max(max_exp_avg_sq, exp_avg_sq, out=
                        max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(y=paddle.to_tensor(
                        group['eps']))
                else:
                    denom = exp_avg_sq.sqrt().add_(y=paddle.to_tensor(group
                        ['eps']))
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2
                    ) / bias_correction1
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state[
                    'step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state[
                    'step']))
                step_size = paddle.full_like(x=denom, fill_value=step_size)
                step_size.divide_(y=paddle.to_tensor(denom)).clip_(min=
                    lower_bound, max=upper_bound).multiply_(y=paddle.
                    to_tensor(exp_avg))
                p.data.add_(-step_size)
        return loss
