require 'torch'
require 'nn'
require 'optim'

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
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 5000,    -- write check point on this interval
  niter = 10000,          -- number of iterations through dataset
  ntrain = math.huge,   -- how big one epoch should be
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
local net
if opt.finetune == '' then -- build network from scratch

  -- SpatialConvolution is (nInputChannels, nOutputChannels, 1, kernelWidth, 1, stride, 0, padding)
  -- the constants are for the other dimension (which is unused)

  net = nn.Sequential()

  net:add(nn.SpatialConvolution(1, 32, 1,64, 1,2, 0,32))
  net:add(nn.SpatialBatchNormalization(32))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(1,8, 1,8))

  net:add(nn.SpatialConvolution(32, 64, 1,32, 1,2, 0,16))
  net:add(nn.SpatialBatchNormalization(64))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(1,8, 1,8))

  net:add(nn.SpatialConvolution(64, 128,  1,16, 1,2, 0,8))
  net:add(nn.SpatialBatchNormalization(128))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(1,8, 1,8))
  --net:add(nn.SpatialDropout(0.5))

  net:add(nn.SpatialConvolution(128, 256,  1,8, 1,2, 0,4))
  net:add(nn.SpatialBatchNormalization(256))
  net:add(nn.ReLU(true))
  --net:add(nn.SpatialDropout(0.5))

  net:add(nn.ConcatTable():add(nn.SpatialConvolution(256, 1000, 1,16, 1,12, 0,4))
                          :add(nn.SpatialConvolution(256,  401, 1,16, 1,12, 0,4)))

  net:add(nn.ParallelTable():add(nn.SplitTable(3)):add(nn.SplitTable(3)))
  net:add(nn.FlattenTable())


  local output_net = nn.ParallelTable()
  for i=1,8 do
    output_net:add(nn.Sequential():add(nn.Contiguous()):add(nn.LogSoftMax()):add(nn.Squeeze()))
  end
  net:add(output_net)

  -- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init) -- loop over all layers, applying weights_init

else -- load in existing network
  print('loading ' .. opt.finetune)
  net = torch.load(opt.finetune)
end

print(net)

-- define the loss
local criterion = nn.ParallelCriterion(false)
for i=1,8 do
  criterion:add(nn.DistKLDivCriterion())
end

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, 1)
local labels = {}
for i=1,opt.label_time_steps do
  labels[i] = torch.Tensor(opt.batchSize, 1000)
end
for i=1,opt.label_time_steps do
  labels[opt.label_time_steps+i] = torch.Tensor(opt.batchSize, 401)
end
local err

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  for i=1,#labels do
    labels[i] = labels[i]:cuda()
  end
  net:cuda()
  criterion:cuda()
end

-- conver to cudnn if needed
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end

-- get a vector of parameters
local parameters, gradParameters = net:getParameters()

-- show graphics
disp = require 'display'
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im,data_label,data_extra
local fx = function(x)
  gradParameters:zero()
  
  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im,data_label,data_label2,data_extra = data:getBatch()
  data_tm:stop()
    
  -- ship data to GPU
  input:copy(data_im:view(opt.batchSize, 1, opt.fineSize, 1))
  for i=1,opt.label_time_steps do
    labels[i]:copy(data_label:select(3,i)) 
  end
  for i=1,opt.label_time_steps do
    labels[opt.label_time_steps+i]:copy(data_label2:select(3,i))
  end
  
  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, labels) / #labels * opt.lambda
  local df_do = criterion:backward(output, labels) 
  for i=1,#labels do df_do[i]:mul(opt.lambda / #labels) end 
  net:backward(input, df_do)

  -- return gradients
  return err, gradParameters
end

local counter = 0
local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train main loop
for epoch = 1,opt.niter do -- for each epoch
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do -- for each mini-batch
    collectgarbage() -- necessary sometimes
    
    tm:reset()

    -- do one iteration
    optim.adam(fx, parameters, optimState)

    -- logging
    if counter % 10 == 0 then
      table.insert(history, {counter, err})
      disp.plot(history, {win=opt.display_id+1, title=opt.name, labels = {"iteration", "err"}})
    
      local w = net.modules[1].weight:clone():float():squeeze()
      disp.image(w, {win=opt.display_id+30, title=("conv1 min: %.4f, max: %.4f"):format(w:min(), w:max())})
    end
    
    counter = counter + 1
    
    print(('%s: Iteration: [%d]\t Time: %.3f  DataTime: %.3f  '
              .. '  Err: %.4f'):format(
            opt.name, counter, 
            tm:time().real, data_tm:time().real,
            err and err or -1))

    -- save checkpoint
    -- :clearState() compacts the model so it takes less space on disk
    if counter % opt.saveIter == 0 then
      print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
      paths.mkdir('checkpoints')
      paths.mkdir('checkpoints/' .. opt.name)
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
      --torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_optim.t7', optimState)
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_history.t7', history)
    end
  end
end
