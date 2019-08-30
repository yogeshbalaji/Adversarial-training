## Code for PGD attacker
import torch
import torch.nn.functional as F


class PGDAttacker(object):

    def __init__(self, attack_eps):
        self.attack_eps = attack_eps

    def attack(self, x, y, net, attack_steps, attack_lr, random_init=True, target=None):
        """
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr: Learning rate of attacker
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :return:
        """

        x_adv = x.clone()

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.attack_eps

        for i in range(attack_steps):
            x_adv.requires_grad = True

            net.zero_grad()
            logits = net(x_adv)

            if target is None:
                # Untargeted attacks - gradient ascent
                loss = F.cross_entropy(logits, y)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv + attack_lr * grad

            else:
                # Targeted attacks - gradient descent
                assert target.size() == y.size()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - attack_lr * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)
            x_adv = x_adv.detach()

        return x_adv






