--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

require 'torch'
require 'audio'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')

-- this function reads in the data files
function dataset:__init(args, tid)
  for k,v in pairs(args) do self[k] = v end

  self.frames = tds.Vec()
  self.videos = tds.Vec()
  self.video_position = tds.Vec()
  self.video_lengths = tds.Vec()

  local last_video = 'first'
  local position = 1
  local lengths = 1
  print('loading ' .. string.format(args.label_text_file, tid))
  for line in io.lines(string.format(args.label_text_file, tid)) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.frames:insert(split[1])

    local cur_video = split[1]:sub(63, -14)
    if cur_video ~= last_video then
      self.videos:insert(cur_video)
      self.video_position:insert(position)

      if last_video ~= 'first' then
        self.video_lengths:insert(lengths)
        assert(lengths <= self.label_time_steps)
      end

      last_video = cur_video
      lengths = 1

    else
      lengths = lengths + 1
    end

    position = position + 1
  end
  self.video_lengths:insert(lengths)
  assert(lengths <= self.label_time_steps)

  assert(#self.video_position == #self.videos)
  assert(#self.video_lengths == #self.videos)

  print('found data for ' .. #self.frames .. ' frames')
  print('found data for ' .. #self.videos .. ' videos')

  self.labels = torch.Tensor(#self.frames, self.label_dim)
  self.labels2 = torch.Tensor(#self.frames, self.label2_dim)

  collectgarbage()
  print('reading ' .. string.format(args.label_binary_file, tid))
  local fd = torch.DiskFile(string.format(args.label_binary_file, tid), "r"):binary()
  self.labels:storage():copy(fd:readFloat(#self.frames * self.label_dim))
  fd:close()

  collectgarbage()
  print('reading ' .. string.format(args.label2_binary_file, tid))
  local fd = torch.DiskFile(string.format(args.label2_binary_file, tid), "r"):binary()
  self.labels2:storage():copy(fd:readFloat(#self.frames * self.label2_dim))
  fd:close()

  collectgarbage()
  collectgarbage()

  print('loaded label tensor of size ' .. string.format('%d x %d', self.labels:size(1), self.labels:size(2)))

  print('loaded label2 tensor of size ' .. string.format('%d x %d', self.labels2:size(1), self.labels2:size(2)))

  -- check the labels
  assert(self.labels:ge(0):all())
  assert(self.labels:le(1):all())
  assert(math.abs(self.labels:sum(2):mean() - 1) < 0.001)

  assert(self.labels2:ge(0):all())
  assert(self.labels2:le(1):all())
  assert(math.abs(self.labels2:sum(2):mean() - 1) < 0.001)

end

function dataset:size()
  return #self.videos
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, labelTable, label2Table, extraTable)
   local data, labels, labels2
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 1)
   data = torch.Tensor(quantity, self.fineSize) 
   scalarLabels = torch.Tensor(quantity, self.label_dim, self.label_time_steps):fill(-1111)
   scalarLabels2 = torch.Tensor(quantity, self.label2_dim, self.label_time_steps):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i]:copy(labelTable[i])
      scalarLabels2[i]:copy(label2Table[i])
   end
   return data, scalarLabels, scalarLabels2, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local labelTable = {}
   local label2Table = {}
   local extraTable = {}
   for i=1,quantity do
      local idx = torch.random(1, #self.videos)

      local out,data_label,data_label2 = self:trainHook(idx)

      table.insert(dataTable, out)
      table.insert(labelTable, data_label)
      table.insert(label2Table, data_label2)
      table.insert(extraTable, self.videos[idx])
   end
   return self:tableToOutput(dataTable,labelTable,label2Table,extraTable)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
   assert(start_idx)
   assert(stop_idx)
   assert(start_idx<stop_idx)
   assert(start_idx<=#self.data)
   local dataTable = {}
   local labelTable = {}
   local extraTable = {}
   for idx=start_idx,stop_idx do
      if idx > #self.data then
        break
      end
      local data_path = self.data_root .. '/' .. self.data[idx]

      local out,data_label = self:trainHook(data_path) 
      table.insert(dataTable, out)
      table.insert(labelTable, data_label)
      table.insert(extraTable, data_path)
   end
   return self:tableToOutput(dataTable,labelTable,extraTable)
end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(idx)
   collectgarbage()

   local video = self.videos[idx] -- video is the filename of the MP4

   local video_pos = self.video_position[idx] -- the starting position into self.labels
   local video_length = self.video_lengths[idx] -- the starting position into self.labels (between 1 and 4)

   local seconds_of_video = video_length * opt.video_frame_time -- in range 1 to 20

   -- load the audio 
   local data_path = self.data_root .. '/' .. video .. '.mp3'
   local input = self:loadAudio(data_path)
   local label = self.labels[{ {video_pos, video_pos+video_length-1}, {} }] 
   local label2 = self.labels2[{ {video_pos, video_pos+video_length-1}, {} }] 

   if input:size(1) < seconds_of_video * opt.sample_rate then
     local orig_size = input:size(1)
     input = input:resize(seconds_of_video * opt.sample_rate)
     input[{ {orig_size, input:size(1)} }]:fill(0)
   end
    
   if video_length < opt.label_time_steps then
     input = input[{ {1, seconds_of_video * opt.sample_rate} }]
     local repeat_times = math.ceil(opt.loadSize / input:size(1))
     input = input:repeatTensor(repeat_times)
     input = input[{ {1, opt.loadSize} }]

     label = label:repeatTensor(opt.label_time_steps, 1)
     label = label[{ {1, opt.label_time_steps}, {} }] 

     label2 = label2:repeatTensor(opt.label_time_steps, 1)
     label2 = label2[{ {1, opt.label_time_steps}, {} }] 

   else 
     input = input[{ {1, opt.loadSize} }]

     assert(video_length == opt.label_time_steps)
   end

   -- transpose
   label = label:t()
   label2 = label2:t()

   -- subtract mean from audio
   input:add(-self.mean)

   assert(input:dim() == 1)
   assert(input:size(1) == opt.fineSize)
   assert(label:dim() == 2)
   assert(label:size(1) == self.label_dim)
   assert(label:size(2) == self.label_time_steps)

   assert(label2:dim() == 2)
   assert(label2:size(1) == self.label2_dim)
   assert(label2:size(2) == self.label_time_steps)

   return input,label,label2
end

-- reads an image disk
-- if it fails to read the image, it will use a blank image
-- and write to stdout about the failure
-- this means the program will not crash if there is an occassional bad apple
function dataset:loadAudio(path)
  local ok,input = pcall(audio.load, path)
  if not ok then
     print('warning: failed loading: ' .. path)
     input = torch.zeros(opt.loadSize) 
  else
    -- if two channels, pick the first channel
    if input:size(2) > 1 then
      input = input:select(2,1) 
    end

    input = input:view(-1) -- flatten to vector
  end

  --local zero_count = input:eq(0):float():mean()
  --if zero_count > 0.1 then
  --  --print('warning: zero percentage is ' .. string.format('%.4f', zero_count) .. ' for ' .. path)
  --end

  input:mul(2^-23)

  assert(input:max() <= 256)
  assert(input:min() >= -256)

  assert(input:dim() == 1)

  return input
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt, tid)
