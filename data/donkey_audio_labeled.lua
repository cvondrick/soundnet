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
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  -- we are going to read args.data_list
  -- we split on the tab
  -- we use tds.Vec() because they have no memory constraint 
  self.data = tds.Vec()
  self.label = tds.Vec()
  for line in io.lines(args.data_list) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    split[2] = tonumber(split[2])+1 -- TODO there is a +1 here, because our files are 0-indexed
    if split[2] ~= nil then 
      self.data:insert(split[1])
      self.label:insert(split[2])
    end
  end

  print('found ' .. #self.data .. ' items')
end

function dataset:size()
  return #self.data
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, scalarTable, extraTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   assert(dataTable[1]:dim() == 1)
   data = torch.Tensor(quantity, self.fineSize) 
   scalarLabels = torch.LongTensor(quantity):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = scalarTable[i]
   end
   return data, scalarLabels, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local labelTable = {}
   local extraTable = {}
   for i=1,quantity do
      local idx = torch.random(1, #self.data)
      local data_path = self.data_root .. '/' .. self.data[idx]
      local data_label = self.label[idx]

      local out = self:trainHook(data_path) 
      table.insert(dataTable, out)
      table.insert(labelTable, data_label)
      table.insert(extraTable, self.data[idx])
   end
   return self:tableToOutput(dataTable,labelTable,extraTable)
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
      local data_label = self.label[idx]

      local out = self:trainHook(data_path) 
      table.insert(dataTable, out)
      table.insert(labelTable, data_label)
      table.insert(extraTable, self.data[idx])
   end
   return self:tableToOutput(dataTable,labelTable,extraTable)
end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(path)
   collectgarbage()
   local input = self:loadAudio(path)
   local iT = input:size(1)

   -- do random crop
   local oT = self.fineSize
   local t1
   t1 = math.ceil(torch.uniform(1e-2, iT-oT))

   out = input[{ {t1, t1 + oT-1} }]

   return out
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

    input = input:repeatTensor(3)
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
trainLoader = dataLoader(opt)
