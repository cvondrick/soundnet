SoundNet
========

Code for paper "[SoundNet: Learning Sound Representations from Unlabeled Video](https://arxiv.org/abs/1610.09001 )" by Yusuf Aytar, Carl Vondrick, Antonio Torralba. NIPS 2016

We learn rich natural sound representations by capitalizing on large amounts of unlabeled sound data collected in the wild. We leverage the natural synchronization between vision and sound to learn an acoustic representation using two-million unlabeled videos. We propose a student-teacher training procedure which transfers discriminative visual knowledge from well established visual models (e.g. ImageNet and PlacesCNN) into the sound modality using unlabeled video as a bridge.

<img src='http://projects.csail.mit.edu/soundnet/soundnet.jpg'>

Visualization of learned conv1 filters:
<img src='http://projects.csail.mit.edu/soundnet/conv1.png'>

Requirements
============
 - torch7
 - torch7 audio (and sox)
 - torch7 hdf5 (only for feature extraction)
 - probably a GPU
 
Pretrained Model
================
We provide pre-trained models that are trained over 2,000,000 unlabeled videos. You can download the 8 layer and 5 layer models [here](http://data.csail.mit.edu/soundnet/soundnet_models_public.zip). We recommend the 8 layer network.

Recognize Categories
====================

You can use SoundNet to recognize sounds or as features (see next section). To recognize objects and scenes, you can use our provided script. First, create a text file where each line lists an audio file you wish to process. We use MP3 files, but most audio formats should be supported. Then, extract predictions into HDF5 files like so:

```bash
$ list=data.txt th extract_predictions.lua
```

where `data.txt` is this text file. It will write HDF5 files to the location of the input files with the scores of each category. To map the dimension index back to the category name, use the files `categories/categories_places2.txt` for scenes and `categories/categories_imagenet.txt` for objects.

The script will also output the top scoring object and scene category. E

Feature Extraction
==================

Using our network, you can extract good features for natural sounds. You can use our provided script to extract features. First, create a text file where each line lists an audio file you wish to process. We use MP3 files, but most audio formats should be supported. Then, extract features into HDF5 files like so:

```bash
$ list=data.txt th extract_feat.lua
```

where `data.txt` is this text file. It will write HDF5 files to the location of the input files with the features. You can then do anything you wish with these features. 
 
By default, it will extract the `conv7` layer. You can extract other layers like so:
 
```bash
$ list=data.txt layer=24 th extract_feat.lua
````
 
Advanced
--------
 
 If you want to write your own feature extraction code, it is very easy in Torch:

```lua
sound = audio.load('file.mp3'):select(2,1):clone():mul(2^-23):view(1,1-1,1):cuda()

net = torch.load('soundnet8_final.t7')
net:forward(sound)
features = net.modules[24].output:float()
```

Finetuning
==========

If you want to fine-tune SoundNet on your own dataset, you can use `main_finetune.lua`. Create a text file that lists your MP3 files and their corresponding categories as integers. Each line lists one file in the format: filename [space] integer. For example:
```
/path/to/file1.mp3 1
/path/to/file2.mp3 5
/path/to/file3.mp3 2
```
The integer at the end of the text file specifies the category. Note they should start counting at 1 (not zero).

Then, you can finetune SoundNet with the command:
```
finetune=models/soundnet8_final.t7 data_list=dataset.txt data_root=/ nClasses=5 name=mynet1 th main_finetune.lua
```
where the `data_list` variable points to your text file you created above, and `nClasses` specifies the number of categories you have.

Note you may have to modify `main_finetune.lua` depending on your needs. This is meant to just get you started. In particular, you may need to adjust the `fineSize` and `loadSize`, which specify how many samples from the waveform to use.

Training
========

The code for training is in `main_train.lua`, which will train the 8 layer SoundNet model. You can also use `main_train_small.lua` to train the 5 layer SoundNet model. To start training, just do:

```bash
$ CUDA_VISIBLE_DEVICES=0 th main_train.lua
```

The code for loading the data is in `data/donkey_audio.lua`. The training code will launch several threads. Each thread reads a different subset of the dataset. It will read MP3 files into a raw waveform. For the labels, it reads a large binary file that stores the class probabilities computed from ImageNet and Places networks.

Data
====

We plan to release 2 million MP3s and their corresponding class probabilities soon. Stay tuned.

References
==========

If you use SoundNet in your research, please cite our paper:

    SoundNet: Learning Sound Representations from Unlabeled Video 
    Yusuf Aytar, Carl Vondrick, Antonio Torralba
    NIPS 2016
