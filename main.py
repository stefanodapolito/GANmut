import os
import argparse
import solver
from torch.backends import cudnn
import dataloader_affectnet
import torch


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    loader = None

    img_root = os.path.join(config.dataset_root, config.imgs_folder)

    train_exp_csv_file = os.path.join(config.dataset_root, "training.csv")

    train_dataset = dataloader_affectnet.DataloaderAffectnet(
        img_size=config.image_size, exp_classes=config.c_dim, is_transform=True
    )
    train_dataset.load_data(train_exp_csv_file, img_root)
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    solve = solver.Solver(loader, config)

    solve.train()

    # Solver for training and testing StarGAN.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument("--image_size", type=int, default=128, help="image resolution")
    parser.add_argument(
        "--g_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of G",
    )
    parser.add_argument(
        "--d_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--g_repeat_num", type=int, default=6, help="number of residual blocks in G"
    )
    parser.add_argument(
        "--d_repeat_num", type=int, default=6, help="number of strided conv layers in D"
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    
    parser.add_argument('--lambda_regularization', type=float, default=10., help="regularization R1 or gradient-penalty" )
    
    parser.add_argument(
        "--regularization_type",
        type=str,
        default="gp",
        choices=["R1", "gp"],
    )
    
    
    
    parser.add_argument(
        "--lambda_d_strength",
        type=float,
        default=1.0,
        help="weight for stregth expr penalty",
    )
    parser.add_argument(
        "--lambda_g_strength",
        type=float,
        default=1.0,
        help="weight for stregth expr penalty",
    )
    parser.add_argument(
        "--lambda_expr",
        type=float,
        default=1.0,
        help="weight for d learning atent coordinat",
    )
    parser.add_argument("--lambda_d_info", type=float, default=1.0)
    parser.add_argument("--lambda_g_info", type=float, default=1.0)
    parser.add_argument("--lambda_prediction", default=0.5, type=float)
    parser.add_argument("--architecture_v2", default =False, type=bool)
    
    # parser.add_argument('--lambda_d_AU',type=float, default=1.)
    # parser.add_argument('--lambda_g_fake_AU',type=float, default=1.)
    # parser.add_argument('--lambda_g_AU',type=float, default=1.)
    # parser.add_argument('--lambda_d_fake_AU',type=float, default=1.)
    # parser.add_argument('--lambda_projector',type=float, default=1.)
    # parser.add_argument('--lambda_g_loss_proj_rec',type=float, default=1.)
    # parser.add_argument('--AU',type=bool,default=False)
    # parser.add_argument('--ACE',type=bool,default=False)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/srv/beegfs02/scratch/emotion_perception/data/csevim/datasets/affectnet",
        help="dataset_root",
    )
    parser.add_argument(
        "--imgs_folder", type=str, default="cropped_align_affectnet68lms"
    )

    parser.add_argument("--tridimensional", default=False, type=bool)
    parser.add_argument(
        "--parametrization",
        default="linear",
        choices=["linear", "gaussian"],
        help="parametrization used, i.e, GANmut or GGANmut",
    )

    # Training configuration.

    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=2000000,
        help="number of total iterations for training D",
    )
    parser.add_argument(
        "--num_iters_decay",
        type=int,
        default=100000,
        help="number of iterations for decaying lr",
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0001, help="learning rate for G"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for D"
    )
    parser.add_argument(
        "--n_critic", type=int, default=5, help="number of D updates per each G update"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--resume_iters", type=int, default=None, help="resume training from this step"
    )
    parser.add_argument(
        "--n_r_l",
        type=int,
        default=5,
        help="number per training batch of codes sampled outside the directions for GANmut, see paper",
    )
    parser.add_argument(
        "--n_r_g",
        type=int,
        default=9,
        help="number per training batch of codes that are not equal to the mean of any emotion mode for GGANmut, see paper",
    )
    parser.add_argument(
        "--cycle_loss",
        type=str,
        default="approximate",
        choices=["approximate", "original"],
    )
    # Test configuration.
    parser.add_argument(
        "--test_iters", type=int, default=200000, help="test model from this step"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "print_axes"]
    )
    parser.add_argument("--use_tensorboard", type=str2bool, default=True)

    # Directories.
    parser.add_argument("--celeba_image_dir", type=str, default="data/celeba/images")
    parser.add_argument(
        "--attr_path", type=str, default="data/celeba/list_attr_celeba.txt"
    )
    parser.add_argument("--rafd_image_dir", type=str, default="data/RaFD/train")
    parser.add_argument("--log_dir", type=str, default="stargan/logs")
    parser.add_argument("--model_save_dir", type=str, default="stargan/models")
    parser.add_argument("--sample_dir", type=str, default="stargan/samples")
    parser.add_argument("--result_dir", type=str, default="stargan/results")

    # Step size.
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--sample_step", type=int, default=20000)
    parser.add_argument("--model_save_step", type=int, default=2000)
    parser.add_argument("--lr_update_step", type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
