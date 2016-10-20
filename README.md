SoundNet
========

Code for paper "SoundNet: Learning Sound Representations from Unlabeled Video" by Yusuf Aytar, Carl Vondrick, Antonio Torralba. NIPS 2016

We learn rich natural sound representations by capitalizing on large amounts of unlabeled sound data collected in the wild. We leverage the natural synchronization between vision and sound to learn an acoustic representation using two-million unlabeled videos. Unlabeled video has the advantage that it can be economically acquired at massive scales, yet contains useful signals about natural sound. We propose a student-teacher training procedure which transfers discriminative visual knowledge from well established visual models (e.g. ImageNet and PlacesCNN) into the sound modality using unlabeled video as a bridge. Our sound representation yields significant performance improvements over the state-of-the-art results on standard benchmarks for acoustic scene/object classification. Visualizations suggest some high-level semantics automatically emerge in the sound network, even though it is trained without ground truth labels. 

<img src='http://web.mit.edu/vondrick/soundnet/soundnet.jpg'>

Requirements
============
 - torch7
 - torch7 audio (and sox)
 - torch7 hdf5 (only for feature extraction)
 - probably a GPU

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
net = torch.load('soundnet8_final.t7')

sound = audio.load('file.mp3')

if sound:size(2) > 1 then sound = sound:select(2,1) end -- select first channel (mono)
sound:mul(2^-23)                                        -- make range [-256, 256]
sound = sound:view(1, 1, -1, 1)                         -- shape to BatchSize x 1 x DIM x 1
sound = sound:cuda()                                    -- ship to GPU
sound = sound:view(1, 1, -1, 1):cuda()

net:forward(sound)
feat = net.modules[24].output:float()
```

Training
========

Coming soon

References
==========

If you use this code in your research, please cite our paper:

    Learning Sound Representations from Unlabeled Video 
    Yusuf Aytar, Carl Vondrick, Antonio Torralba
    NIPS 2016
