require 'torch'
require 'nn'
require 'optim'
require 'dpnn'
require 'cunn'
require 'cudnn'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'audio_labeled',   -- indicates what dataset load to use (in data.lua)
  nThreads = 16,        -- how many threads to pre-fetch data
  batchSize = 256,      -- self-explanatory
  loadSize = 22050*5,       -- when loading images, resize first to this size
  fineSize = 22050*5,       -- crop this size from the loaded image 
  lr = 0.001,           -- learning rate
  lambda = 250,
  nClasses = 50,
  beta1 = 0.9,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 20,    -- write check point on this interval
  niter = 50,          -- number of iterations through dataset
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  finetune = 'models/soundnet8_final.t7', 
  name = 'soundnet_ft',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  display_port = 8001,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  data_root = '/data/vision/torralba/crossmodal/soundnet/ESC-50',
  data_list = '/data/vision/torralba/crossmodal/soundnet/ESC-50/splits/train1.txt',
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

print('loading ' .. opt.finetune)
local net = torch.load(opt.finetune)

print('modifying net')
for i=1,4 do net:remove(#net.modules) end
net:add(nn.SpatialConvolution(1024, opt.nClasses, 1,4, 1,1, 0,0))
net:add(nn.View(opt.nClasses):setNumInputDims(3))

print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, 1)
local labels = torch.Tensor(opt.batchSize)
local err

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  labels = labels:cuda()
  net:cuda()
  criterion:cuda()
end

-- conver to cudnn if needed
-- if this errors on you, you can disable, but will be slightly slower
if opt.gpu > 0 and opt.cudnn > 0 then
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
  data_im,data_label = data:getBatch()
  data_tm:stop()
    
  -- ship data to GPU
  input:copy(data_im:view(opt.batchSize, 1, opt.fineSize, 1))
  labels:copy(data_label)

  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, labels)
  local df_do = criterion:backward(output, labels) 
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
