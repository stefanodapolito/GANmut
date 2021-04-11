import dlib
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import cv2
from imutils.face_utils import rect_to_bb
import os
import numpy as np

from model_linear_2d import Generator as Generator_l2
from model_gaussian_2d import Generator as Generator_g2


class LinearModel:

    def __init__(self, G_path_l2='../models/lin_2d/1000000-G.ckpt', g_conv_dim=64, c_dim=7, g_repeat_num=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)
        self.G_l2 = Generator_l2(self.device, g_conv_dim, c_dim, g_repeat_num)
        self.G_l2.load_state_dict(torch.load(G_path_l2, map_location=lambda storage, loc: storage))
        self.G_l2.to(self.device)
        self.detector = dlib.get_frontal_face_detector()

    def emotion_edit(self, img_path, theta, rho, save=False):
        img = cv2.imread(img_path, 1)  # BGR
        img_rgb = img[:, :, [2, 1, 0]]
        plt.title('Original Image')
        plt.imshow(img_rgb)

        # extract face
        det = self.detector(img, 1)[0]
        (x, y, w, h) = rect_to_bb(det)
        face = cv2.resize(img[y:y + h, x:x + w], (128, 128))

        plt.figure()
        plt.title('Detected face')
        plt.imshow(face[:, :, [2, 1, 0]])

        # adapt image format for G
        face = face.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
        face = (face / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        face = torch.from_numpy(face).float().unsqueeze(0).to(self.device)

        # edit emotion
        mode = 'manual_selection'
        expr = (torch.tensor([np.cos(theta), np.sin(theta)]) * rho).to(self.device).float()
        face_g = self.G_l2(face, None, None, mode=mode, manual_expr=expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5
        face_g = face_g.transpose(0, 2).transpose(0, 1).detach().cpu().numpy()

        plt.figure()
        plt.title('Edited face')
        plt.imshow(face_g)

        # insert edited face in original image
        img_rgb[y:y + h, x:x + w] = cv2.resize(face_g, (h, w)) * 255

        plt.figure()
        plt.title('Edited image')
        plt.imshow(img_rgb)

        if save:
            save_dir = "../edited_images"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            img_name = f'theta_{theta}_rho_{rho}' + os.path.split(img_path)[-1]
            img_name = os.path.join(save_dir, img_name)
            plt.imsave(img_name, img_rgb)
