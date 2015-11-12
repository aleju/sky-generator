require 'torch'
require 'nn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'

local models = {}

-- Returns a G for given dimensions and cuda mode.
-- @param dimensions Table of image dimensions, i.e. {channels, height, width}.
-- @param noiseDim Currently ignored.
-- @param cuda Whether to activate cuda mode.
-- @returns Sequential
function models.create_G(dimensions, cuda)
    return models.create_G_a(dimensions, cuda)
end

-- G to upsample a 32x32 image in color mode.
function models.create_G_a(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(cudnn.SpatialConvolutionUpsample(dimensions[1]+1, 64, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(64, 64, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(64, 128, 5, 5, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(128, dimensions[1], 7, 7, 1))
    inner:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))
    model_G:add(inner)
    if cuda then
        model_G:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end
    
    model_G = require('weight-init')(model_G, 'heuristic')
    
    if cuda then
        model_G:get(3):cuda()
    end
    
    return model_G
end

-- Returns a D for given dimensions and cuda mode.
-- @param dimensions Table of image dimensions, i.e. {channels, height, width}.
-- @param cuda Whether to activate cuda mode.
-- @returns Sequential
function models.create_D(dimensions, cuda)
    return models.create_D_a(dimensions, cuda)
end

-- Creates a D for upsampling images to 32x32 grayscale.
function models.create_D_a(dimensions, cuda)
    local model_D = nn.Sequential()
    
    model_D:add(nn.CAddTable())
    if cuda then
        model_D:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    
    inner:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialMaxPooling(2, 2))
    inner:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialMaxPooling(2, 2))
    inner:add(nn.Dropout())
    
    inner:add(nn.View(128 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    inner:add(nn.Linear(128 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 128))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(128, 1))
    inner:add(nn.Sigmoid())
    model_D:add(inner)
    if cuda then
        model_D:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end

    model_D = require('weight-init')(model_D, 'heuristic')

    if cuda then
        model_D:get(3):cuda()
    end

    return model_D
end

return models
