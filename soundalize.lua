require 'torch'
require 'nn'
require 'optim'
require 'dpnn'
require 'cunn'
require 'cudnn'
require 'audio'
require 'image'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'audio',   -- indicates what dataset load to use (in data.lua)
  nThreads = 40,        -- how many threads to pre-fetch data
  batchSize = 64,      -- self-explanatory
  loadSize = 22050*20,       -- when loading images, resize first to this size
  fineSize = 22050*20,       -- crop this size from the loaded image 
  lr = 0.001,           -- learning rate
  lambda = 250,
  beta1 = 0.9,          -- momentum term for adam
  niter = 1000,          -- number of iterations through dataset
  layer = 21,
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  name = 'soundnet',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  display_port = 8001,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  data_root = '/data/vision/torralba/crossmodal/flickr_videos/soundnet/mp3',
  label_binary_file = '/data/vision/torralba/crossmodal/soundnet/features/VGG16_IMNET_TRAIN_B%04d/prob',
  label2_binary_file = '/data/vision/torralba/crossmodal/soundnet/features/VGG16_PLACES2_TRAIN_B%04d/prob',
  label_text_file = '/data/vision/torralba/crossmodal/soundnet/lmdbs/train_frames4_%04d.txt',
  label_dim = 1000,
  label2_dim = 401,
  label_time_steps = 4,
  video_frame_time = 5, -- 5 seconds
  sample_rate = 22050,
  mean = 0,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- define the model
local net = torch.load('models/soundnet8_final.t7')

print(net)
print('extracting layer ' .. opt.layer)

while #net.modules > opt.layer do net:remove(#net.modules) end

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, 1)

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  net:cuda()
end

-- conver to cudnn if needed
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end

local feats 
local files = {}

for i=1,opt.niter do

  data_tm:reset(); data_tm:resume()
  data_im,data_label,data_label2,data_extra = data:getBatch()
  data_tm:stop()
    
  -- ship data to GPU
  input:copy(data_im:view(opt.batchSize, 1, opt.fineSize, 1))
  local output = net:forward(input):squeeze()

  if i == 1 then
    feats = torch.zeros(opt.niter, opt.batchSize, output:size(2), output:size(3))

    for j=1,#net.modules do
      local tmp = net.modules[j].output:squeeze()
      print('Layer ' .. j .. ' has size: ' .. tmp:size(2) .. 'x' .. tmp:size(3))
    end
  end

  xlua.progress(i, opt.niter)

  feats[i]:copy(output)
  for j=1,opt.batchSize do table.insert(files, data_extra[j]) end
end

feats = feats:view(-1, feats:size(3), feats:size(4))

for neuron=1,feats:size(2) do
  print(neuron)

  local sig = feats:select(2,neuron)
  local scores = torch.max(sig, 2)

  local thresh = sig:maskedSelect(sig:gt(0))
  if thresh:dim() > 0 then
    thresh = thresh:median()[1]

    local _, idx = torch.sort(scores, 1, true)
    idx = idx:view(-1)

    local activations = {}

    for j=1,10 do
      local input = audio.load(opt.data_root .. files[idx[j]] .. '.mp3')
      if input:size(2) > 1 then
        input = input:select(2,1) 
      end
      input = input:view(-1)

      local repeat_times = math.ceil(opt.loadSize / input:size(1))
      input = input:repeatTensor(repeat_times)
      input = input[{ {1, opt.loadSize} }]

      local sig_sel = sig[idx[j]]:gt(thresh):float()
      sig_scale = image.scale(sig_sel:view(-1,1), 1, input:size(1), 'simple')
      sig_scale = sig_scale:view(-1):gt(0)

      table.insert(activations, input:maskedSelect(sig_scale))
    end

    local activations = torch.cat(activations, 1):view(-1,1)

    audio.save("soundalize/" .. string.format('%03d.mp3', neuron), activations, opt.sample_rate)

  end
end
