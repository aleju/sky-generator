require 'torch'
require 'nn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'layers.UnPooling'

local models = {}

-- Returns a G for given dimensions and cuda mode.
-- @param dimensions Table of image dimensions, i.e. {channels, height, width}.
-- @param noiseDim Currently ignored.
-- @param cuda Whether to activate cuda mode.
-- @returns Sequential
function models.create_G(dimensions, cuda)
    return models.create_G_e(dimensions, cuda)
end

-- G to upsample a 32x32 image in color mode.
function models.create_G_a(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(cudnn.SpatialConvolutionUpsample(3+1, 32, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(32, 32, 5, 5, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(32, 128, 7, 7, 1))
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

function models.create_G_b(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    local concat = nn.Concat(2)
    local s1 = nn.Sequential()
    s1:add(nn.Select(2, 1))
    s1:add(nn.Reshape(1, dimensions[2], dimensions[3]))
    local s2 = nn.Sequential()
    s2:add(nn.Select(2, 2))
    s2:add(nn.Reshape(1, dimensions[2], dimensions[3]))
    concat:add(s1)
    concat:add(s2)
    model_G:add(concat)
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(nn.SpatialConvolution(1+1, 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    inner:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    local size = 32 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    inner:add(nn.View(size))
    inner:add(nn.Linear(size, 256))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(256, dimensions[1] * dimensions[2] * dimensions[3]))
    inner:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))
    --inner:add(nn.UnPooling(2))
    --inner:add(cudnn.SpatialConvolutionUpsample(3, 3, 3, 3, 1))
    
    model_G:add(inner)
    if cuda then
        model_G:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end
    
    model_G = require('weight-init')(model_G, 'heuristic')
    
    if cuda then
        inner:cuda()
    end
    
    return model_G
end

function models.create_G_e(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    local concat = nn.Concat(2)
    local s1 = nn.Sequential()
    s1:add(nn.Select(2, 1))
    s1:add(nn.Reshape(1, dimensions[2], dimensions[3]))
    local s2 = nn.Sequential()
    s2:add(nn.Select(2, 2))
    s2:add(nn.Reshape(1, dimensions[2], dimensions[3]))
    concat:add(s1)
    concat:add(s2)
    model_G:add(concat)
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(nn.SpatialConvolution(1+1, 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    inner:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    local size = 32 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    inner:add(nn.View(size))
    inner:add(nn.Linear(size, 512))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(512, dimensions[1] * dimensions[2] * dimensions[3]))
    --inner:add(nn.Tanh())
    --inner:add(nn.MulConstant(2))
    inner:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))
    --inner:add(nn.UnPooling(2))
    --inner:add(cudnn.SpatialConvolutionUpsample(3, 3, 3, 3, 1))
    
    model_G:add(inner)
    if cuda then
        model_G:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    end
    
    model_G = require('weight-init')(model_G, 'heuristic')
    
    if cuda then
        inner:cuda()
    end
    
    return model_G
end

function models.create_G_c(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(cudnn.SpatialConvolutionUpsample(3+1, 32, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    inner:add(cudnn.SpatialConvolutionUpsample(32, 128, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(128, 128, 5, 5, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(128, 16, 5, 5, 2))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(16, 3, 5, 5, 1))
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

-- G to upsample a 32x32 image in color mode.
function models.create_G_d(dimensions, cuda)
    local model_G = nn.Sequential()
    
    model_G:add(nn.JoinTable(2, 2))
    
    if cuda then
        model_G:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    inner:add(cudnn.SpatialConvolutionUpsample(3+1, 64, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(64, 128, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(128, 256, 3, 3, 1))
    inner:add(nn.PReLU())
    inner:add(cudnn.SpatialConvolutionUpsample(256, dimensions[1], 11, 11, 1))
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
    return models.create_D_c(dimensions, cuda)
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
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialMaxPooling(2, 2))
    
    inner:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    --inner:add(nn.SpatialMaxPooling(2, 2))
    inner:add(nn.Dropout())
    
    local size = 256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    inner:add(nn.View(size))
    inner:add(nn.Linear(size, 512))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(512, 1))
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

-- Creates a D for upsampling images to 32x32 grayscale.
function models.create_D_b(dimensions, cuda)
    local model_D = nn.Sequential()
    
    model_D:add(nn.CAddTable())
    if cuda then
        model_D:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    
    inner:add(nn.SpatialConvolution(dimensions[1], 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    --inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    --inner:add(nn.Dropout())
    
    --local size = 32 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    local size = 32 * 0.25 * dimensions[2] * dimensions[3]
    inner:add(nn.View(size))
    inner:add(nn.Linear(size, 256))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(256, 1))
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

-- Creates a D for upsampling images to 32x32 grayscale.
function models.create_D_c(dimensions, cuda)
    local model_D = nn.Sequential()
    
    model_D:add(nn.CAddTable())
    if cuda then
        model_D:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    end
    
    local inner = nn.Sequential()
    
    inner:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    --inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, (5-1)/2))
    inner:add(nn.PReLU())
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    inner:add(nn.Dropout())
    
    --local size = 32 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    local size = 64 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    inner:add(nn.View(size))
    inner:add(nn.Linear(size, 1024))
    inner:add(nn.PReLU())
    inner:add(nn.Linear(1024, 1))
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
