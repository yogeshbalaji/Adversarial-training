import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import os.path as osp
import torch.backends.cudnn as cudnn
from attacks import PGDAttacker
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, args):

        self.args = args

        transformer = T.Compose([
            T.ToTensor()
        ])
        kwargs = {'num_workers': 4, 'pin_memory': True}

        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, transform=transformer, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        self.model = models.WRN(depth=32, width=10, num_classes=10)
        self.model = torch.nn.DataParallel(self.model).cuda()

        # Loading model
        assert self.args.restore is not None

        model_data = torch.load(self.args.restore)
        self.model.load_state_dict(model_data['model'])
        self.model.eval()

        cudnn.benchmark = True
        self.attacker = PGDAttacker(args.attack_eps)
        self.save_path = self.args.save_path

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def visualize(self):
        """
        Module to compute and visualize adversarial perturbations
        """

        num_vis = self.args.num_vis
        input_list = []
        input_adv_list = []

        count = 0
        for (i, batch) in enumerate(self.train_loader):
            imgs, labels = batch
            input = imgs.cuda()
            labels = labels.cuda()

            input_adv = self.attacker.attack(input, labels, self.model, self.args.attack_steps, self.args.attack_lr,
                                         random_init=True)

            input_np = input.cpu().numpy()
            input_adv_np = input_adv.cpu().numpy()

            input_list.append(input_np)
            input_adv_list.append(input_adv_np)

            count += input_np.shape[0]

            if count > num_vis:
                break

        input_list = np.vstack(input_list)
        input_adv_list = np.vstack(input_adv_list)

        input_list = input_list.transpose(0, 2, 3, 1)
        input_list = (input_list * 255.0).astype(np.uint8)
        input_adv_list = input_adv_list.transpose(0, 2, 3, 1)
        input_adv_list = (input_adv_list * 255.0).astype(np.uint8)

        ## Generating visualization
        fig, axs = plt.subplots(2, num_vis, figsize=(15, 7))

        for i in range(num_vis):
            axs[0][i].imshow(input_list[i])

        for i in range(num_vis):
            axs[1][i].imshow(input_adv_list[i])

        plt.savefig('{}/adv.png'.format(self.args.save_path))
