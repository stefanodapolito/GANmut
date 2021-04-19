import os
import time
import datetime
import sys

from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np

import models.model_linear_2d
import models.model_gaussian_2d
import models.model_linear_3d
import models.model_gaussian_3d


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_regularization = config.lambda_regularization
        self.regularization_type = config.regularization_type
        self.lambda_d_strength = config.lambda_d_strength
        self.lambda_g_strength = config.lambda_g_strength
        self.lambda_g_info = config.lambda_g_info
        self.lambda_d_info = config.lambda_d_info
        self.tridimensional = config.tridimensional
        self.parametrization = config.parametrization
        self.lambda_expr = config.lambda_expr
        self.lambda_prediction = config.lambda_prediction
        self.architecture_v2 = config.architecture_v2

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iter = config.resume_iter
        self.n_r_l = config.n_r_l
        self.n_r_g = config.n_r_g
        self.cycle_loss = config.cycle_loss

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Used device: ", self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # losses
        self.KLDivLoss = torch.nn.KLDivLoss(reduction="batchmean")
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)
        self.Softmax = torch.nn.Softmax(dim=1)

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        if self.parametrization == "linear":
            
            if self.tridimensional:

                self.G = models.model_linear_3d.Generator(
                    self.device,
                    self.g_conv_dim,
                    self.c_dim,
                    self.g_repeat_num,
                    n_r=self.n_r_l,
                )
                self.D = models.model_linear_3d.Discriminator(
                    self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num
                )
                
            else:
                
                self.G = models.model_linear_2d.Generator(
                    self.device,
                    self.g_conv_dim,
                    self.c_dim,
                    self.g_repeat_num,
                    n_r=self.n_r_l,
                )
                self.D = models.model_linear_2d.Discriminator(
                    self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num
                )
                
                
        elif self.parametrization == "gaussian":
            
            if self.tridimensional:

                self.G = models.model_gaussian_3d.Generator(
                    self.device, self.g_conv_dim, self.c_dim, self.g_repeat_num
                )
                self.D = models.model_gaussian_3d.Discriminator(
                    self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num
                )
                
            else:
                
                self.G = models.model_gaussian_3d.Generator(
                    self.device, self.g_conv_dim, self.c_dim, self.g_repeat_num
                )
                self.D = models.model_gaussian_3d.Discriminator(
                    self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num
                )

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iter):
        """Restore the trained generator and discriminator."""
        print("Loading the trained models from step {}...".format(resume_iter))
        G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(resume_iter))
        D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(resume_iter))
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage)
        )
        self.D.load_state_dict(
            torch.load(D_path, map_location=lambda storage, loc: storage)
        )

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from utils.logger import Logger

        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    # Copied from StarGAN v2 code
    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert grad_dout2.size() == x_in.size()
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""

        c_trg_list = []
        for i in range(c_dim):

            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute cross entropy loss."""

        return F.cross_entropy(logit, target)

    def train(self):
        """Train GANmut."""
        # Set data loader.
        data_loader = self.loader

        # Fetch fixed inputs for debugging.

        data_iter = iter(data_loader)

        x_fixed, c_org, _ = next(data_iter)

        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)
        label_emotions = torch.tensor(
            [em for em in range(self.c_dim)], device=self.device, dtype=torch.long
        )

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iter = 0
        if self.resume_iter:
            start_iter = self.resume_iter
            self.restore_model(self.resume_iter)

        # Start training.
        print("Start training...")
        start_time = time.time()
        for i in range(start_iter, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.

            try:
                x_real, label_org, _ = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org, _ = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            if self.parametrization == "linear":
                # expression strength of all expressions except neutral set into the interval [0.2,1]
                expression_strength = (
                    torch.rand(x_real.size(0), device=self.device) * 0.8 + 0.2
                )

                # neutral expression strength set to 0
                expression_strength[label_trg.eq(0)] = (
                    0.2 * (expression_strength[label_trg.eq(0)] - 0.2) / 0.8
                )
                neutral_mask = (expression_strength > 0.2).to(torch.float)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                # Compute loss with real images.
                x_real.requires_grad_()
                out_src, out_cls, _ = self.D(x_real)

                d_loss_real = -torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org)

                # Compute loss with fake images.
                x_fake, cord = self.G(x_real, c_trg, expression_strength)
                out_src_fake, out_cls, cord_hat = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src_fake)
                d_loss_info = F.mse_loss(cord_hat, cord)

                # Compute regularization loss
                if self.regularization_type == "gp":
                    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                    x_hat = (
                        alpha * x_real.data + (1 - alpha) * x_fake.data
                    ).requires_grad_(True)
                    out_src, _, _ = self.D(x_hat)
                    d_loss_regularization = self.gradient_penalty(out_src, x_hat)
                elif self.regularization_type == "R1":
                    d_loss_regularization = self.r1_reg(out_src, x_real)
                else:
                    sys.exit("Regularization not supported")

                # Backward and optimize.
                d_loss = (
                    d_loss_real
                    + d_loss_fake
                    + self.lambda_cls * d_loss_cls
                    + self.lambda_regularization * d_loss_regularization
                    + self.lambda_d_info * d_loss_info
                )
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss["D/loss_real"] = d_loss_real.item()
                loss["D/loss_fake"] = d_loss_fake.item()
                loss["D/loss_cls"] = d_loss_cls.item()
                loss["D/loss_regularization"] = d_loss_regularization.item()
                loss["D/loss_info"] = d_loss_info.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    with torch.no_grad():
                        _, cls_real, cord_hat_real = self.D(x_real)

                    x_fake, cord = self.G(x_real, c_trg, expression_strength)
                    out_src, out_cls, cord_hat = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls[5:], label_trg[5:])
                    expr_strength_hat = (F.softmax(out_cls, dim=1)).max(1)[0]

                    g_loss_expression_strength = (
                        F.mse_loss(
                            expr_strength_hat * neutral_mask,
                            expression_strength * neutral_mask,
                            reduction="sum",
                        )
                        / torch.sum(neutral_mask)
                    )
                    g_loss_info = torch.nn.functional.mse_loss(cord_hat, cord)

                    # Target-to-original domain..shape

                    if self.cycle_loss == "approximate":
                        # if True:
                        expr_strength_real = (F.softmax(cls_real, dim=1)).max(1)[0]
                        x_reconst, _ = self.G(x_fake, c_org, expr_strength_real)
                    elif self.cycle_loss == "original":
                        x_reconst, _ = self.G(
                            x_fake,
                            c_org,
                            None,
                            mode="manual_selection",
                            manual_expr=cord_hat_real,
                        )
                    else:
                        sys.exit(
                            "cycle loss can be either 'approximate' either 'original' "
                        )

                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = (
                        g_loss_fake
                        + self.lambda_rec * g_loss_rec
                        + self.lambda_cls * g_loss_cls
                        + self.lambda_g_strength * g_loss_expression_strength
                        + self.lambda_g_info * g_loss_info
                    )
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss["G/loss_fake"] = g_loss_fake.item()
                    loss["G/loss_rec"] = g_loss_rec.item()
                    loss["G/loss_cls"] = g_loss_cls.item()
                    loss["G/loss_expr_strength"] = g_loss_expression_strength.item()
                    loss["G/loss_info"] = g_loss_info.item()

            if self.parametrization == "gaussian":

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                x_real.requires_grad_()
                out_src, out_cls = self.D(x_real)

                d_loss_real = -torch.mean(out_src)
                d_loss_cls = self.classification_loss(
                    out_cls[:, 0 : self.c_dim], label_org
                )

                # Compute loss with fake images.
                x_fake, label_trg, expr = self.G(x_real)
                out_src_fake, out_cls = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src_fake)
                d_loss_expr = F.mse_loss(out_cls[:, self.c_dim :], expr)

                if self.regularization_type == "gp":
                    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                    x_hat = (
                        alpha * x_real.data + (1 - alpha) * x_fake.data
                    ).requires_grad_(True)
                    out_src, _ = self.D(x_hat)
                    d_loss_regularization = self.gradient_penalty(out_src, x_hat)
                elif self.regularization_type == "R1":
                    d_loss_regularization = self.r1_reg(out_src, x_real)
                else:
                    sys.exit("Regularization not supported")

                # Backward and optimize.
                d_loss = (
                    d_loss_real
                    + d_loss_fake
                    + self.lambda_cls * d_loss_cls
                    + self.lambda_regularization * d_loss_regularization
                    + self.lambda_expr * d_loss_expr
                )

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss["D/loss_real"] = d_loss_real.item()
                loss["D/loss_fake"] = d_loss_fake.item()
                loss["D/loss_cls"] = d_loss_cls.item()
                loss["D/loss_reguarization"] = d_loss_regularization.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    with torch.no_grad():
                        _, expr_org = self.D(x_real)

                    x_fake, mahalanobis_distances, _ = self.G(x_real)
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)

                    probabilities_trg = self.Softmax(-mahalanobis_distances)
                    log_probabilities_trg = self.LogSoftmax(-mahalanobis_distances)

                    g_loss_cls = self.KLDivLoss(
                        self.LogSoftmax(out_cls[:, 0 : self.c_dim]), probabilities_trg
                    ) + self.KLDivLoss(
                        log_probabilities_trg, self.Softmax(out_cls[:, 0 : self.c_dim])
                    )

                    g_loss_prediction = self.classification_loss(
                        out_cls[0 : self.c_dim, 0 : self.c_dim], label_emotions
                    )
                    # Target-to-original domain..shape

                    x_reconst, _, _ = self.G(x_fake, expr_org[:, self.c_dim:])
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    g_loss = (
                        g_loss_fake
                        + self.lambda_rec * g_loss_rec
                        + self.lambda_cls * g_loss_cls
                        + self.lambda_prediction * g_loss_prediction
                    )
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss["G/loss_fake"] = g_loss_fake.item()
                    loss["G/loss_rec"] = g_loss_rec.item()
                    loss["G/loss_cls"] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:

                elapsed_time = time.time() - start_time
                elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    elapsed_time, i + 1, self.num_iters
                )
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            if (i + 1) % self.sample_step == 0:

                with torch.no_grad():
                    if self.parametrization == "linear":

                        print("AXES:")
                        print(self.G.print_axes())

                        expression_str = 0.9 * torch.ones(
                            x_fixed.size(0), dtype=torch.float, device=self.device
                        )
                        x_fake_list = [x_fixed]
                        for c_fixed in c_fixed_list:

                            x_fake_list.append(
                                self.G(
                                    x_fixed,
                                    c_fixed[: x_fixed.size(0)],
                                    expression_str,
                                    "test",
                                )[0][:, [2, 1, 0], :, :]
                            )

                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(
                            self.sample_dir, "{}-images.jpg".format(i + 1)
                        )
                        save_image(
                            self.denorm(x_concat.data.cpu()),
                            sample_path,
                            nrow=1,
                            padding=0,
                        )
                        print(
                            "Saved real and fake images into {}...".format(sample_path)
                        )

                        for emotion in range(1, self.c_dim):

                            c_emotion = c_fixed_list[emotion]
                            x_fake_list = [x_fixed]

                            for strength in range(0, 11):

                                x_fake_list.append(
                                    self.G(
                                        x_fixed,
                                        c_emotion[: x_fixed.size(0)],
                                        strength * 0.1,
                                        "test",
                                    )[0][:, [2, 1, 0], :, :]
                                )

                            x_concat = torch.cat(x_fake_list, dim=3)
                            sample_path = os.path.join(
                                self.sample_dir, f"{i+1}-images_emotion_{emotion}.jpg"
                            )
                            save_image(
                                self.denorm(x_concat.data.cpu()),
                                sample_path,
                                nrow=1,
                                padding=0,
                            )
                            print(
                                "Saved real and fake images into {}...".format(
                                    sample_path
                                )
                            )

                    if self.parametrization == "gaussian":

                        print("MODES:")
                        print(self.G.print_expr())

                        x_fake_list = [x_fixed]
                        for expression in range(self.c_dim):

                            x_fake_list.append(
                                self.G(
                                    x_fixed,
                                    self.G.mu.weight[expression]
                                    .unsqueeze(0)
                                    .repeat(x_fixed.size(0), 1),
                                )[0][:, [2, 1, 0], :, :]
                            )

                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(
                            self.sample_dir, "{}-images.jpg".format(i + 1)
                        )
                        save_image(
                            self.denorm(x_concat.data.cpu()),
                            sample_path,
                            nrow=1,
                            padding=0,
                        )
                        print(
                            "Saved real and fake images into {}...".format(sample_path)
                        )

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(i + 1))
                D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))
