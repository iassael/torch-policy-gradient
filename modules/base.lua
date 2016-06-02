--
--    Copyright (c) 2014, Facebook, Inc.
--    All rights reserved.
--
--    This source code is licensed under the Apache 2 license found in the
--    LICENSE file in the root directory of this source tree. 
--

tools = {}

function tools.transfer(x)
    if opt.cuda and type(x) ~= 'number' then
        return x:cuda()
    else
        return x
    end
end

function tools.Tensor(...)
    local a
    if opt.cuda then
        a = torch.CudaTensor(...)
    else
        a = torch.Tensor(...)
    end
    return a
end

-- Similar to maxParamNorm, but norm is global to Module for which 
-- this is called. Unless moduleLocal is true, in which case, the
-- norm constraint is applied to the norm of all parameters in each
-- component (non-container) module.
function tools.gradParamClip(cur_module, cutoffNorm, moduleLocal)
   if cutoffNorm <= 0 then
      return
   end
   
   local norm = 0
   if moduleLocal and cur_module.modules then
      for i,module in ipairs(cur_module.modules) do
         norm = norm + math.pow(tools.gradParamClip(module, maxOutNorm, maxInNorm), 2)
      end
      norm = math.sqrt(norm)
   else
      local params, gradParams = cur_module:parameters()
      if not (params and gradParams) then
         return norm
      end
      for k,gradParam in pairs(gradParams) do -- pairs for sparse params
         norm = norm + math.pow(gradParam:norm(),2)
      end
      norm = math.sqrt(norm)
      if norm > cutoffNorm then
         -- rescale gradParams to obtain desired cutoffNorm
         for k,gradParam in pairs(gradParams) do
            gradParam:mul(cutoffNorm/norm)
         end
      end
   end
   return norm
end

function tools.dc(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[tools.dc(orig_key)] = tools.dc(orig_value)
        end
        setmetatable(copy, tools.dc(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function g_standardize(vector, mean, standard_deviation)
   local nObservations = vector:size(1)

   if mean == nil then
      mean = torch.mean(vector)
   end

   if standard_deviation == nil then
      local differences = vector - mean
      local squared_differences = torch.cmul(differences, differences)
      local variance = torch.sum(squared_differences) / nObservations
      standard_deviation = math.sqrt(variance)
   end

   local standardized = torch.div(vector - mean, standard_deviation)
   return standardized, mean, standard_deviation
end

function g_destandarize(vector, mean, standard_deviation)
    return vector:clone():mul(standard_deviation):add(mean)
end

function g_model_evaluate(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_model_evaluate)
        end
        return
    end
    if node.__typename ~= nil then
    --if string.match(node.__typename, "Dropout") or string.match(node.__typename, "BatchNormalization") then
        node.train = false
    end
end

function g_model_training(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_model_training)
        end
        return
    end
    if node.__typename ~= nil then
    --if string.match(node.__typename, "Dropout") or string.match(node.__typename, "BatchNormalization") then
        node.train = true
    end
end

function g_cloneGraph(net)
    local params, gradParams = net:parameters()
    if params == nil then
        params = {}
    end
    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i =1,#paramsNoGrad do
            cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
    end
    
    collectgarbage()
    
    mem:close()
    return clone
end

function tools.cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    if params == nil then
        params = {}
    end
    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        local cloneParamsNoGrad
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        if paramsNoGrad then
            cloneParamsNoGrad = clone:parametersNoGrad()
            for i =1,#paramsNoGrad do
                cloneParamsNoGrad[i]:set(paramsNoGrad[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function g_init_gpu(args)
    local gpuidx = args
    gpuidx = gpuidx[1] or 1
    print(string.format("Using %s-th gpu", gpuidx))
    cutorch.setDevice(gpuidx)
    g_make_deterministic(1)
end

function g_make_deterministic(seed)
    torch.manualSeed(seed)
    cutorch.manualSeed(seed)
    torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
    assert(#to == #from)
    for i = 1, #to do
        to[i]:copy(from[i])
    end
end

function g_f2(f)
    return string.format("%.2f", f)
end

function g_f3(f)
    return string.format("%.3f", f)
end

function g_f4(f)
    return string.format("%.4f", f)
end

function g_f5(f)
    return string.format("%.5f", f)
end

function g_f6(f)
    return string.format("%.6f", f)
end

function g_d(f)
    return string.format("%d", torch.round(f))
end




--- other functions

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end



function g_combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function g_create_batch(dataset)
    local batches = {}
    
    -- shuffle at each epoch
    local shuffle = torch.randperm(dataset.x:size(1)):long()
    
    for t = 1,dataset.x:size(1),opt.batch_size do

        -- create mini batch
        local batch_x = {}
        local batch_flow = {}
        local batch_u = {}
        local batch_y = {}
        for i = t,math.min(t+opt.batch_size-1,dataset.x:size(1)) do
            local idx = shuffle[i]
            if idx-1 >= 1 and idx+1 <= dataset.x:size(1) then
                -- load new sample
                local x_prev = dataset.x[idx-1]
                local x_cur = dataset.x[idx]
                local x_next = dataset.x[idx+1]
                local u = dataset.u[idx]

                table.insert(batch_x, torch.cat(x_prev, x_cur))
                table.insert(batch_u, u)
                table.insert(batch_y, torch.cat(x_cur, x_next))
            end
        end

        table.insert(batches, {batch_x, batch_u, batch_y})
    end
    
    dataset.batch = batches
end