import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import os.path as osp
import torch.backends.cudnn as cudnn
from attacks import PGDAttacker
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial


class Visualizer:
    def __init__(self, args):

        self.args = args

        transformer = T.Compose([
            T.ToTensor()
        ])
        kwargs = {'num_workers': 4, 'pin_memory': True}

        train_set = datasets.CIFAR10(args.data_root, train=True, transform=transformer, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.train_samples_np = train_set.data.astype(np.float32)
        self.train_samples_np = self.train_samples_np.transpose(0, 3, 1, 2)
        self.train_samples_np = np.reshape(self.train_samples_np, (self.train_samples_np.shape[0], -1))

        self.train_samples_np = self.train_samples_np / 255.0
        self.labels_np = np.array(train_set.targets)

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

    def get_nn(self, inputs, labels):

        samples_nn = []
        labels_nn = []

        for i in range(inputs.size(0)):
            img_query = inputs[i].cpu().numpy()
            img_query = img_query.astype(np.float32)
            img_query = np.reshape(img_query, (1, -1))

            label_query = labels[i].cpu().numpy()

            valid_indices = (self.labels_np != label_query).nonzero()[0]
            valid_samples = self.train_samples_np[valid_indices]

            dist_mat = scipy.spatial.distance.cdist(valid_samples, img_query)
            dist_mat = np.reshape(dist_mat, (-1))
            min_ind = np.argmin(dist_mat)
            sample_ind = valid_indices[min_ind]
            samples_nn.append(self.train_samples_np[sample_ind])
            labels_nn.append(self.labels_np[sample_ind])

        samples_nn = np.array(samples_nn)
        samples_nn = np.reshape(samples_nn, (samples_nn.shape[0], 3, 32, 32))
        labels_nn = np.array(labels_nn)

        return samples_nn, labels_nn


    def visualize(self):
        """
        Module to compute and visualize adversarial perturbations
        """

        num_vis = self.args.num_vis
        input_list = []
        input_adv_list = []
        input_nn_list = []

        count = 0
        for (i, batch) in enumerate(self.train_loader):
            imgs, labels = batch
            input = imgs.cuda()
            labels = labels.cuda()

            # Random targets
            # target = labels + torch.randint(low=1, high=10, size=labels.size()).cuda()
            # target = torch.fmod(target, 10)

            # Nearest neighbor targets
            inputs_nn, labels_nn = self.get_nn(imgs, labels)
            labels_nn = torch.from_numpy(labels_nn).long().cuda()
            input_nn_list.append(inputs_nn)

            input_adv = self.attacker.attack(input, labels, self.model, self.args.attack_steps, self.args.attack_lr,
                                             random_init=True, target=labels_nn)

            input_np = input.cpu().numpy()
            input_adv_np = input_adv.cpu().numpy()

            input_list.append(input_np)
            input_adv_list.append(input_adv_np)

            count += input_np.shape[0]

            if count > num_vis:
                break

        input_list = np.vstack(input_list)
        input_adv_list = np.vstack(input_adv_list)
        input_nn_list = np.vstack(input_nn_list)

        input_list = input_list.transpose(0, 2, 3, 1)
        input_list = (input_list * 255.0).astype(np.uint8)
        input_adv_list = input_adv_list.transpose(0, 2, 3, 1)
        input_adv_list = (input_adv_list * 255.0).astype(np.uint8)
        input_nn_list = input_nn_list.transpose(0, 2, 3, 1)
        input_nn_list = (input_nn_list * 255.0).astype(np.uint8)

        ## Generating visualization
        fig, axs = plt.subplots(3, num_vis, figsize=(15, 7))

        for i in range(num_vis):
            axs[0][i].imshow(input_list[i])
            axs[0][i].axis('off')

        for i in range(num_vis):
            axs[1][i].imshow(input_adv_list[i])
            axs[1][i].axis('off')

        for i in range(num_vis):
            axs[2][i].imshow(input_nn_list[i])
            axs[2][i].axis('off')

        plt.savefig('{}/adv.png'.format(self.args.save_path))
