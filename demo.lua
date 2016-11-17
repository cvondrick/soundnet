require 'torch'
require 'nn'
require 'audio'
require 'cunn'
require 'cudnn'

-- set good defaults
torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

-- load the network
local net = torch.load('models/soundnet8_final.lua')
net:cuda()

print('Network:')
print(net)

-- load data
local sound = audio.load('demo.mp3')

-- data preprocessing
if sound:size(2) > 1 then sound = sound:select(2,1):clone() end -- select first channel (mono)
sound:mul(2^-23)                                        -- make range [-256, 256]
sound = sound:view(1, 1, -1, 1)                         -- shape to BatchSize x 1 x DIM x 1
sound = sound:cuda()                                    -- ship to GPU

-- forward pass
net:forward(sound)

-- extract layer
local feat = net.modules[10].output
