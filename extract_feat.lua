require 'torch'
require 'nn'
require 'audio'
require 'hdf5'
require 'cunn'
require 'cudnn'

opt = {
  model = "models/soundnet8_final.t7",   -- which soundnet model to load
  list = "",                             -- text file listing mp3's to process, one per line
  layer = 24,                            -- which layer to extract features at
  force=0,                               -- force overwrite
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.list == '' then error('you must specify text file of audio files to process') end

-- set good defaults
torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

-- load the network
print('Loading network: ' .. opt.model)
local net = torch.load(opt.model) 
net:cuda()

print('Network:')
print(net)

print('Extracting layer ' .. opt.layer)

-- remove unnecessary layers
while #net.modules ~= opt.layer do net:remove(#net.modules) end

net:evaluate()

-- http://stackoverflow.com/questions/4990990/lua-check-if-a-file-exists
function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

for line in io.lines(opt.list) do
  local out_file = line .. '.soundnet.h5'

  if file_exists(out_file) and opt.force ~= 1 then
    print('Skip ' .. out_file)

  else
    print('Read ' .. line)
    local sound = audio.load(line)

    -- data preprocessing
    if sound:size(2) > 1 then sound = sound:select(2,1):clone() end -- select first channel (mono)
    sound:mul(2^-23)                                        -- make range [-256, 256]
    sound = sound:view(1, 1, -1, 1)                         -- shape to BatchSize x 1 x DIM x 1
    sound = sound:cuda()                                    -- ship to GPU

    -- forward pass
    net:forward(sound)

    -- extract layer
    local feat = net.modules[opt.layer].output:float()

    print('Write ' .. out_file)
    local fd = hdf5.open(out_file, 'w')
    fd:write('layer' .. opt.layer, feat:squeeze())
    fd:close()
  end

  net:clearState()
end
