SoundNet
========


Usage
-----

  net = torch.load('soundnet8_final.t7')
  sound = audio.load('demo.mp3')
  
  sound = sound:view(1, 1, -1, 1):cuda()
  net:forward(sound)

  feat = net.modules[10].output:float()
