## GANmut - Official PyTorch Implementation

### Accepted to CVPR 2021

> **GANmut: Learning Interpretable Conditional Space for Gamut of Emotions**<br>
> Stefano d'Apolito<sup>1</sup>, Danda Pani Paudel<sup>1</sup>, Zhiwu Huang<sup>1</sup>, Andres Romero<sup>1</sup>, Luc Van Gool<sup>1,2</sup>    <br/>
> <sup>1</sup>Computer Vision Lab, ETH Zurich, Switzerland, <sup>2</sup>PSI, KU Leuven, Belgium <br>
> 
> <br>
>
> **Abstract:** *Humans can communicate emotions through a plethora
of facial expressions, each with its own intensity, nuances
and ambiguities. The generation of such variety by means
of conditional GANs is limited to the expressions encoded in
the used label system. These limitations are caused either
due to burdensome labelling demand or the confounded label space. On the other hand, learning from inexpensive
and intuitive basic categorical emotion labels leads to limited emotion variability. In this paper, we propose a novel
GAN-based framework that learns an expressive and interpretable conditional space (usable as a label space) of emotions, instead of conditioning on handcrafted labels. Our
framework only uses the categorical labels of basic emotions to learn jointly the conditional space as well as emotion manipulation. Such learning can benefit from the image variability within discrete labels, especially when the
intrinsic labels reside beyond the discrete space of the defined. Our experiments demonstrate the effectiveness of the
proposed framework, by allowing us to control and generate a gamut of complex and compound emotions while using only the basic categorical emotion labels during training.*

## Dependencies
`pip install -r requirments.txt` to install required dependencies. Finally, install also Pytorch (code has been tested with version 1.4) and Torchvision.

## Editing facial emotions with pre-trained networks
To see an example of how to use GANmut to edit face emotion, install jupyter and open `notebook/images_generation.ipynb`.


## Training networks

#### Dataset 
GANmut requires a large and varied training dataset, such as [AffectNet](https://ieeexplore.ieee.org/abstract/document/8013713?casa_token=2m7z--0nVk8AAAAA:fA4dfo5o8U0pPazaqLMnkwZh_jVTpA0kFsU3MURM5viMLNiCLA_OSLep7uCUzQrHc0H381Q). To train it with a custom dataset place the images in the folder `dataset/imgs`, and add the 
file `dataset/training.csv`. This should have as header `subDirectory_filePath,expression`, and each row should contain the name of the images (e.g., `imgs/Einstein.jpg`), and its discrete emotion label, (e.g., *0* for Neutral, *1* for Happy and so on). See a mini example in `dataset` folder.

### Important Training Parameters<a name="params"></a>
Some of the configuraions in  scripts "train_ImageNet.sh" and "train_cifar.sh" need to be set according to your experiments. Some of the important parameters are:
| Parameter | Description
| :---- | :----------
| --c_dim | The number of different categorical emotion in the training set
| --parametrization | Whether to use the linear or gaussian model
| --resume_iter | From which iteration resume the training (The relative checkpoint must be preset)
| --model_save_step | Every model_save_step iterations the model is saved


 Other parameters are described in n "main.py"


#### Launch training
```bash
# Train GANmut with dataset in in the folder dataset
python -m core.main  --dataset_root dataset --imgs_folder imgs --image_size 128 --c_dim 7    
                    --sample_dir samples/samples_linear_2d --log_dir GANmut/logs_linear_2d 
                    --model_save_dir GANmut/models_linear_2d --result_dir GANmut/results_linear_2d

# Test StarGAN using the CelebA dataset
python  -m core.main  --dataset_root dataset --imgs_folder imgs --image_size 128  --parametrization gaussian 
                           --c_dim 7  --lambda_cls=2. --sample_dir samples/samples_gaussian_2d --log_dir GANmut/logs_gaussian_2d 
                           --model_save_dir GANmut/models_gaussian_2d --result_dir GANmut/results_gaussian_2d 
```

##  Contact<a name="Contact"></a>
For any questions, suggestions, or issues with the code, please contact Stefano at <a>stefanodapolito1305@gmail.com</a>.

## How to Cite<a name="How-to-Cite"></a>
If you find this project helpful, please consider citing us as follows:
```bash
@inproceedings{sdapolito2021GANmut,
      title = {GANmut: Learning Interpretable Conditional Space for Gamut of Emotions},
      author    = {d'Apolito, Stefano and
                   Paudel, ‪Danda Pani and
                   Huang, Zhiwu and
                   Romero, Andres and
                   Van Gool, Luc},
      year = {2021},
      booktitle = {2021 {IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2021}
}
```

## Acknowledgements

The code has been developed through an extended modification of [yunjey's StarGAN](https://github.com/yunjey/stargan)








