require 'torch'
require 'nn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'

local models = {}

-- Creates the encoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_encoder16(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.View(64 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(64 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Linear(1024, noiseDim))
    --model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')
  
    return model
end

-- Creates the encoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_encoder32(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    
    model:add(nn.View(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Linear(1024, noiseDim))
    --model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')
  
    return model
end

-- Creates the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder(dimensions, noiseDim)
    local imgSize = dimensions[1] * dimensions[2] * dimensions[3]
  
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 1024))
    model:add(nn.PReLU())
    model:add(nn.Linear(1024, imgSize))
    model:add(nn.Sigmoid())
    model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-16px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling16(dimensions, noiseDim)
    local imgSize = dimensions[1] * dimensions[2] * dimensions[3]
  
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 128*4*8))
    model:add(nn.View(128, 4, 8))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())
    
    --model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-32px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling32(dimensions, noiseDim)
    local imgSize = dimensions[1] * dimensions[2] * dimensions[3]
  
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 64*8*16))
    model:add(nn.View(64, 8, 16))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.PReLU(nil, nil, true))
    
    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())
    
    --model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim)
    if dimensions[2] == 16 then
        return models.create_G_decoder_upsampling16(dimensions, noiseDim)
    else
        return models.create_G_decoder_upsampling32(dimensions, noiseDim)
    end
end

-- Creates the G as an autoencoder (encoder+decoder).
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_autoencoder(dimensions, noiseDim)
    local model = nn.Sequential()
    
    if dimensions[2] == 16 then
        model:add(models.create_G_encoder16(dimensions, noiseDim))
    else
        model:add(models.create_G_encoder32(dimensions, noiseDim))
    end
    
    if dimensions[2] == 16 then
        model:add(models.create_G_decoder_upsampling16(dimensions, noiseDim))
    else
        model:add(models.create_G_decoder_upsampling32(dimensions, noiseDim))
    end
    
    return model
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_D(dimensions, cuda)
    if dimensions[2] == 16 then
        return models.create_D16b(dimensions, cuda)
    else
        return models.create_D32e(dimensions, cuda)
    end
end

function models.create_D16(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(1024 * (1/4)*(1/4) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(1024 * (1/4)*(1/4) * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D16b(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout(0.2))
    
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    
    conv:add(nn.View(128 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(128 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout())
    
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32b(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout())
    
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 512, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(512, 512, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(512 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(512 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32c(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout())
    
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(256, 256, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32d(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    --conv:add(nn.Dropout())
    
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32e(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    
    conv:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout())
    
    conv:add(nn.View(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(256 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- Creates V.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @returns nn.Sequential
function models.create_V(dimensions)
    if dimensions[2] == 16 then
        return models.create_V16(dimensions)
    else
        return models.create_V32(dimensions)
    end
end

function models.create_V16(dimensions)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialDropout(0.2))
  
    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialDropout())
    
    local imgSize = 0.25 * 0.25 * dimensions[2] * dimensions[3]
    model:add(nn.View(256 * imgSize))
  
    model:add(nn.Linear(256 * imgSize, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 2))
    model:add(nn.SoftMax())
  
    model = require('weight-init')(model, 'heuristic')
  
    return model
end

function models.create_V32(dimensions)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.Dropout())
  
    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialDropout())
    local imgSize = 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]
    model:add(nn.View(256 * imgSize))
  
    model:add(nn.Linear(256 * imgSize, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 2))
    model:add(nn.SoftMax())
  
    model = require('weight-init')(model, 'heuristic')
  
    return model
end

return models
