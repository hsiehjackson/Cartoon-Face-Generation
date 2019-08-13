# [107-2] Applied Deep Learning - Cartoon Face Generation
In this project, we are trying to implement supervised conditional generation on [CartoonSet](https://google.github.io/cartoonset/index.html) dataset. The simple method is to use [ACGAN](https://arxiv.org/pdf/1610.09585.pdf) structure with adversarial loss and auxiliary loss jointly. Moreover, we applied different GAN loss and useful techniques to improve the performance.  


## Best Image Results
<img src="https://i.imgur.com/TKJmULF.gif" alt="drawing" /> 

## Usage
### Training
* Git clone the code and install package
```
git clone https://github.com/hsiehjackson/
pip install -r requirements
```
* Download files and extract zip file
```
bash download.sh
unzip download.zip
```
* Train ACGAN with different model (default=concat and SN)
```
python src/acgan_train.py [folder_name] --generator_type=concat || noconcat --discriminator_type=noSN || SN || SNPJ
```

* Train ACGAN with different loss (default=WGGP)
```
python src/acgan_train.py [folder_name] --loss_type=MM || NS || WGCP || WGGP || WGDIV
```
* Files Saved
```
./saves/ (folder for all the files we save)
./saves/[folder_name]/models/ (folder for model checkpoints)
./saves/[folder_name]/train_sample/ (folder for some sample images when training)
./saves/[folder_name]/plot.json (model training procedure)
```

### Testing

* Test Your ACGAN
```
python src/acgan_test.py ./saves/[folder_name] [epoch_num] --seed=[YOUR SEED]
```
* Test My Best ACGAN
```
python src/acgan_best.py  --seed=[YOUR SEED]
```

* Test FID Score
```
cd test/FID_evaluation
python run_fid.py ../../saves/[folder_name]/test_images/ep-[epoch-num]/
```
* Files Saved
```
./saves/[folder_name]/test_sample/ (sample images with specific model checkpoints)
./saves/[folder_name]/test_images/ep-[epoch_num]/ (folder for test FID images with specific model checkpoints)
```
### Others
* Plot training progress
```
python src/plot.py ./saves/[folder_name]/plot.json
```

## Dataset Introdution
The original dataset has lots of attributes including 10 artwork, 4 color, and 4 proportion, which may be too complicated to learn. Therefore, we use the preprocessed images with small size and only 4 attributes, such as hair/eye/face color and w/wo glasses. The sample image is shown in the following and the label for each attribute is an one-hot vector.
<img src="https://i.imgur.com/MaCIZKg.jpg" alt="drawing"/> 

## Baseline Framework
Our baseline network is ACGAN, shown in the following. However, we applied several techniques to help training GAN. Besides real images and one-hot condition, we alse need gaussian noise for the whole training procedure.

For the auxiliary loss, we use binary cross entropy loss, which we simply concatenate all one-hot encoding as the 1D condition. For the adversarial loss, we use several WGAN tricks.


| Generator  | 
| :--------: |
| <img src="https://i.imgur.com/nTNYR6W.png" alt="drawing"/> |

| Discriminator  | 
| :--------: |
| <img src="https://i.imgur.com/3bM78Wp.png" alt="drawing"/> |


## Techniques for Training GAN
> ### Model Condition on Label
With concatenation conditions, generator has a better ability to generate specific images. The concatenation would prevent the condition information from disappearing.

| Generator  with hidden concatenation conditions| 
| :--------: |
| <img src="https://i.imgur.com/LAfK063.png" alt="drawing"/> |

Unlike concatenation with condition, using projection can enable the discriminator to only use specific condition information to determine real/fake. This method may be more powerful because each conditions has different features to tell the real/fake.

| Discriminator with conditions projection for adversarial loss| 
| :--------: |
| <img src="https://i.imgur.com/QTRFMiO.png" alt="drawing"/> |

> ### Spectral Normalization
Previous studies had showed spectral normalzation is so powerful to **reduce mode collapse** problems. We remove all the batch normalization layer and add spectral normalization layer after each convolution and linear layer on discriminator. 

> ### Techniques for Wasserstein Distance
There are several tricks on Wasserstein Distance to make the training procedure more stable. We implement WG-CLIP, WG-DIV, and WG-GP to show the difference performance. Our default setting is  WG-GP.

## Training Procedure Results
The best loss results for discriminator are higher fake loss and lower real loss while generator are both lower adversarial and auxillary loss.

From the following results, we can find generator with hidden concatenation condition give a more stable auxillary loss.

| G without hidden condition| G with hidden condition|
| :--------: | :--------: |
| <img src="https://i.imgur.com/3k6DgVH.png" alt="drawing" height="130"/> |<img src="https://i.imgur.com/cwt4RCr.png" alt="drawing" height="130"/> |

With spectral normalization, we can obtain an impressive result, which is stable and without any explosion. However, the initial procedure of projection methods may see some disturbance due to the difficulty to learn specific condition information for adversarial loss.

| D with SN layer | D with SN layer + projection|
| :--------: | :--------: |
| <img src="https://i.imgur.com/bGolb7D.png" alt="drawing" height="130"/> |<img src="https://i.imgur.com/Vk43EV1.png" alt="drawing" height="130"/> |

Using clipping techniques, we can see a stable but easily-converged result which may limit the learning procedure. Considering divergence techniques, it is better than clipping but with more disturbance. 

| WG-CLIP | WG-DIV |
| :--------: | :--------: |
| <img src="https://i.imgur.com/PtIaG6h.png" alt="drawing" height="130"/> |<img src="https://i.imgur.com/psBkOiF.png" alt="drawing" height="130"/> |

## FID Results

The default setting for our stucture is using generator hidden concatenation condition and discriminator WG-GP adversarial loss.
| | Default | without hidden | WG Clip | WG DIV | SN | SN+Proj |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|Epoch|300|200|400|500|450|800|
|FID↓|89|131|216|55|62|**44**|

