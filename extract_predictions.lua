require 'torch'
require 'nn'
require 'audio'
require 'hdf5'
require 'cunn'
require 'cudnn'

opt = {
  model = "models/soundnet8_final.t7",   -- which soundnet model to load
  list = "",                             -- text file listing mp3's to process, one per line
  write=0,
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
for i=1,3 do net:remove(#net.modules) end
net:add(nn.MapTable():add(cudnn.SpatialSoftMax())) 
net:cuda()

print('Network:')
print(net)

net:evaluate()

-- http://stackoverflow.com/questions/4990990/lua-check-if-a-file-exists
function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

-- read in categories
local places_cats = {}
local imagenet_cats = {}
for line in io.lines('categories/categories_places2.txt') do table.insert(places_cats, line) end
for line in io.lines('categories/categories_imagenet.txt') do table.insert(imagenet_cats, line) end

local min_length = 10*22050

for line in io.lines(opt.list) do
  local out_file = line .. '.soundnet_categories.h5'

  if file_exists(out_file) and opt.force ~= 1 then
    print('Skip ' .. out_file)

  else
    local sound = audio.load(line)

    -- data preprocessing
    if sound:size(2) > 1 then sound = sound:select(2,1):clone() end -- select first channel (mono)
    sound:mul(2^-23)                                        -- make range [-256, 256]
    sound = sound:view(1, 1, -1, 1)                         -- shape to BatchSize x 1 x DIM x 1

    if sound:size(3) < min_length then
      sound = sound:repeatTensor(1,1,math.ceil(min_length/sound:size(3)),1)
    end

    sound = sound:cuda()                                    -- ship to GPU

    -- forward pass
    local feat = net:forward(sound)

    if opt.write then
      local fd = hdf5.open(out_file, 'w')
      fd:write('object', feat[1]:float()) 
      fd:write('scene', feat[2]:float()) 
      fd:close()
    end

    local mid_idx = math.ceil(feat[1]:size(3)/2)
    local _, imagenet_idx = feat[1]:float():select(3, mid_idx):squeeze():max(1)
    local _, places_idx = feat[2]:float():select(3, mid_idx):squeeze():max(1)

    print(('Video: %s  Object: %s  Scene: %s'):format(line, imagenet_cats[imagenet_idx[1]], places_cats[places_idx[1]]))
  end

  net:clearState()
end
