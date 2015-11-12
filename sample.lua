require 'torch'
require 'image'
require 'paths'
require 'pl'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'stn'
require 'LeakyReLU'
NN_UTILS = require 'utils.nn_utils'
DATASET = require 'dataset'

OPT = lapp[[
    --save_base     (default "logs")                          Directory in which the base 16x16 networks are stored.
    --save_c2f22    (default "logs")                          Directory in which the 16 to 22 coarse to fine networks are stored.
    --save_c2f32    (default "logs")                          Directory in which the 22 to 32 coarse to fine networks are stored.
    --G_base        (default "et1b_adversarial.net")          Filename for the 16x16 base network to load G from.
    --D_base        (default "et1b_adversarial.net")          Filename for the 16x16 base network to load D from.
    --G_c2f22       (default "adversarial_c2f_16_to_22.net")  Filename for the 16 to 22 coarse to fine network to load D from.
    --D_c2f22       (default "adversarial_c2f_16_to_22.net")  Filename for the 16 to 22 coarse to fine network to load D from.
    --G_c2f32       (default "adversarial_c2f_22_to_32.net")  Filename for the 22 to 32 coarse to fine network to load D from.
    --D_c2f32       (default "adversarial_c2f_22_to_32.net")  Filename for the 22 to 32 coarse to fine network to load D from.
    --neighbours                                              Whether to search for nearest neighbours of generated images in the dataset (takes long)
    --scale         (default 16)                              Height/Width of images in the base network.
    --grayscale                                               Grayscale mode on/off
    --writeto       (default "samples")                       Directory to save the images to
    --seed          (default 1)                               Random number seed to use.
    --gpu           (default 0)                               GPU to run on
    --runs          (default 1)                               How often to sample and save images
    --noiseDim      (default 100)                             Noise vector size (for 16x16 net).
    --batchSize     (default 16)                              Sizes of batches.
    --aws                                                     Run in AWS mode.
]]

-- Paths I used during testing stages, just ignore them
--[[
    --save_base     (default "logs/final")                 directory in which the networks are saved
    --save_c2f22    (default "logs/final")
    --save_c2f32    (default "logs/final")
    --G_base        (default "et1b_adversarial.net")      
    --D_base        (default "et1b_adversarial.net")      
    --G_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --D_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --G_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --D_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --neighbours                                           Whether to search for nearest neighbours of generated images in the dataset (takes long)
    --scale         (default 16)
    --grayscale                                            grayscale mode on/off
    --writeto       (default "samples")                    directory to save the images to
    --seed          (default 1)
    --gpu           (default 0)                            GPU to run on
    --runs          (default 1)                           How often to sample and save images
    --noiseDim      (default 100)
    --batchSize     (default 16)
    --aws                                                  run in AWS mode
]]

if OPT.gpu < 0 then
    print("[ERROR] Sample script currently only runs on GPU, set --gpu=x where x is between 0 and 3.")
    exit()
end

-- Start GPU mode
print("Starting gpu support...")
require 'cutorch'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(OPT.gpu + 1)

-- initialize seeds
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
cutorch.manualSeed(OPT.seed)

-- Image dimensions
if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.scale, OPT.scale}
else
    IMG_DIMENSIONS = {3, OPT.scale, OPT.scale}
end

-- Initialize dataset
DATASET.nbChannels = IMG_DIMENSIONS[1]
DATASET.setFileExtension("jpg")
DATASET.setScale(OPT.scale)

if OPT.aws then
    DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
else
    DATASET.setDirs({"dataset/out_aug_64x64"})
end

-- Main function that runs the sampling
function main()
    -- Load all models
    local G, D, G_c2f22, D_c2f22, G_c2f32, D_c2f32, normMean, normStd = loadModels()
    
    -- We need these global variables for some methods. Ugly code.
    MODEL_G = G
    MODEL_D = D
    
    print("Sampling...")
    for run=1,OPT.runs do
        -- save 64 randomly selected images from the training set
        local imagesTrainList = DATASET.loadRandomImages(64)
        local imagesTrain = torch.Tensor(#imagesTrainList, imagesTrainList[1]:size(1), imagesTrainList[1]:size(2), imagesTrainList[1]:size(3))
        for i=1,#imagesTrainList do
            imagesTrain[i] = imagesTrainList[i]:clone()
        end
        image.save(paths.concat(OPT.writeto, string.format('trainset_s1_%04d_base.jpg', run)), toGrid(imagesTrain, 8))
        
        -- sample 1024 new images from G
        local images = NN_UTILS.createImages(1024, false)
        
        -- validate image dimensions
        if images[1]:size(1) ~= IMG_DIMENSIONS[1] or images[1]:size(2) ~= IMG_DIMENSIONS[2] or images[1]:size(3) ~= IMG_DIMENSIONS[3] then
            print("[WARNING] dimension mismatch between images generated by base G and command line parameters, --grayscale falsly on/off or --scale not set correctly")
            print("Dimension G:", images[1]:size())
            print("Settings:", IMG_DIMENSIONS)
        end
        
        -- save a big image of those 1024 random images
        image.save(paths.concat(OPT.writeto, string.format('random1024_%04d_base.jpg', run)), toGrid(images, 32))
        
        -- Collect the best and worst images (according to D) from these images
        -- Save: 64 best images, 64 worst images, 64 randomly selected images
        local imagesBest, predictions = NN_UTILS.sortImagesByPrediction(images, false, 64)
        local imagesWorst, predictions = NN_UTILS.sortImagesByPrediction(images, true, 64)
        local imagesRandom = selectRandomImagesFrom(images, 64)
        imagesBest = imageListToTensor(imagesBest)
        imagesWorst = imageListToTensor(imagesWorst)
        imagesRandom = imageListToTensor(imagesRandom)
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_base.jpg', run)), toGrid(imagesBest, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_base.jpg', run)), toGrid(imagesWorst, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_base.jpg', run)), toGrid(imagesRandom, 8))
        
        -- Extract the 16 best images and find their closest neighbour in the training set
        if OPT.neighbours then
            local searchFor = {}
            for i=1,16 do
                table.insert(searchFor, imagesBest[i]:clone())
            end
            local neighbours = findClosestNeighboursOf(searchFor)
            image.save(paths.concat(OPT.writeto, string.format('best_%04d_neighbours_base.jpg', run)), toNeighboursGrid(neighbours, 8))
        end
        
        -- Run coarse to fine step for 16px to 22px
        -- [1] upscale, [2] normalize, [3] sharpen/c2f, [4] save
        imagesTrain = upscale(imagesTrain, 22, 22)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s2_up22_%04d.jpg', run)), toGrid(imagesTrain, 8))
        imagesBest = upscale(imagesBest, 22, 22)
        imagesWorst = upscale(imagesWorst, 22, 22)
        imagesRandom = upscale(imagesRandom, 22, 22)
        imagesTrain = normalize(imagesTrain, normMean, normStd)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s3_up22norm_%04d.jpg', run)), toGrid(imagesTrain, 8))
        imagesBest = normalize(imagesBest, normMean, normStd)
        imagesWorst = normalize(imagesWorst, normMean, normStd)
        imagesRandom = normalize(imagesRandom, normMean, normStd)
        local imagesTrainC2F22 = c2f(imagesTrain, G_c2f22, D_c2f22)
        local imagesBestC2F22 = c2f(imagesBest, G_c2f22, D_c2f22)
        local imagesWorstC2F22 = c2f(imagesWorst, G_c2f22, D_c2f22)
        local imagesRandomC2F22 = c2f(imagesRandom, G_c2f22, D_c2f22)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s4_c2f_22_%04d.jpg', run)), toGrid(imagesTrainC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_c2f_22.jpg', run)), toGrid(imagesBestC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_c2f_22.jpg', run)), toGrid(imagesWorstC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_c2f_22.jpg', run)), toGrid(imagesRandomC2F22, 8))
        
        -- Run coarse to fine step for 22px to 32px
        -- [1] upscale, [2] sharpen/c2f, [3] save
        -- Images are already normalized from the previous step
        imagesTrainC2F22 = upscale(imagesTrainC2F22, 32, 32)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s5_up32_%04d_base.jpg', run)), toGrid(imagesTrainC2F22, 8))
        imagesBestC2F22 = upscale(imagesBestC2F22, 32, 32)
        imagesWorstC2F22 = upscale(imagesWorstC2F22, 32, 32)
        imagesRandomC2F22 = upscale(imagesRandomC2F22, 32, 32)
        local imagesTrainC2F32 = c2f(imagesTrainC2F22, G_c2f32, D_c2f32)
        local imagesBestC2F32 = c2f(imagesBestC2F22, G_c2f32, D_c2f32)
        local imagesWorstC2F32 = c2f(imagesWorstC2F22, G_c2f32, D_c2f32)
        local imagesRandomC2F32 = c2f(imagesRandomC2F22, G_c2f32, D_c2f32)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s6_c2f_32_%04d.jpg', run)), toGrid(imagesTrainC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_c2f_32.jpg', run)), toGrid(imagesBestC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_c2f_32.jpg', run)), toGrid(imagesWorstC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_c2f_32.jpg', run)), toGrid(imagesRandomC2F32, 8))
        
        xlua.progress(run, OPT.runs)
    end
    
    print("Finished.")
end

-- Searches for the closest neighbours (2-Norm/torch.dist) for each image in the given list.
-- @param images List of image tensors.
-- @returns List of tables {image, closest neighbour's image, distance}
function findClosestNeighboursOf(images)
    local result = {}
    local trainingSet = DATASET.loadImages(0, 9999999)
    for i=1,#images do
        local img = images[i]
        local closestDist = nil
        local closestImg = nil
        for j=1,trainingSet:size() do
            local dist = torch.dist(trainingSet[j], img)
            if closestDist == nil or dist < closestDist then
                closestDist = dist
                closestImg = trainingSet[j]:clone()
            end
        end
        table.insert(result, {img, closestImg, closestDist})
    end
    
    return result
end

-- Runs the coarse to fine / sharpening on upscaled images.
-- @param images Tensor of images.
-- @param G Coarse to fine generator.
-- @param D Coarse to fine discriminator.
-- @returns Tensor of sharpened images.
function c2f(images, G, D)
    local fineSize = images[1]:size(2)
    -- We let G generate several refinements per image and pick the one that D likes most
    local triesPerImage = 10
    local result = {}
    
    -- Run over each image and fine the best refinement
    for i=1,images:size(1) do
        local imgTensor = torch.Tensor(triesPerImage, images[1]:size(1), fineSize, fineSize)
        local img = images[i]:clone()
        local height = img:size(2)
        local width = img:size(3)
        
        for j=1,triesPerImage do
            imgTensor[j] = img:clone()
        end
        
        local noiseInputs = torch.Tensor(triesPerImage, 1, fineSize, fineSize)
        noiseInputs:uniform(-1, 1)
        
        -- Generate refinements
        local diffs = G:forward({noiseInputs, imgTensor})
        
        -- Rate each refinement
        local predictions = D:forward({diffs, imgTensor})
        
        -- Pick best rated refinement
        local maxval = nil
        local maxdiff = nil
        for j=1,triesPerImage do
            if maxval == nil or predictions[j][1] > maxval then
                maxval = predictions[j][1]
                maxdiff = diffs[j]
            end
        end
        
        -- Conert blurry image + best refinement to the final image
        local imgRefined = torch.add(img, maxdiff)
        imgRefined = torch.clamp(imgRefined, -1.0, 1.0)
        
        table.insert(result, imgRefined)
    end
    
    return imageListToTensor(result)
end

-- Upscale a tensor of images to new height and width.
-- @param images Tensor of images.
-- @param newHeight Desired height of image
-- @param newWidth Desired width of image
-- @returns Tensor of images
function upscale(images, newHeight, newWidth)
    local newImages = torch.Tensor(images:size(1), images[1]:size(1), newHeight, newWidth)
    for i=1,images:size(1) do
        newImages[i] = image.scale(images[i], newHeight, newWidth)
    end
    return newImages
end

-- Convert a list/table of images to a tensor.
-- @param images Table of images.
-- @returns Tensor of images
function imageListToTensor(images)
    local newImages = torch.Tensor(#images, images[1]:size(1), images[1]:size(2), images[1]:size(3))
    for i=1,#images do
        newImages[i] = images[i]
    end
    return newImages
end

-- Normalizes a tensor of images.
-- Currently that projects an images from 0.0 to 1.0 to range -1.0 to +1.0.
-- @param images Tensor of images
-- @param mean_ Currently ignored.
-- @param std_ Currently ignored.
-- @returns images Normalized images (NOTE: images are normalized in-place)
function normalize(images, mean_, std_)
    -- normalizes in-place
    NN_UTILS.normalize(images, mean_, std_)
    return images
end

-- Converts images to one image grid with set amount of rows.
-- @param images Tensor of images
-- @param nrow Number of rows.
-- @return Tensor
function toGrid(images, nrow)
    return image.toDisplayTensor{input=images, nrow=nrow}
end

-- Converts a table of images as returned by findClosestNeighboursOf() to one image grid.
-- @param imagesWithNeighbours Table of (image, neighbour image, distance)
-- @returns Tensor
function toNeighboursGrid(imagesWithNeighbours)
    local img = imagesWithNeighbours[1][1]
    local imgpairs = torch.Tensor(#imagesWithNeighbours*2, img:size(1), img:size(2), img:size(3))
    
    local imgpairs_idx = 1
    for i=1,#imagesWithNeighbours do
        imgpairs[imgpairs_idx] = imagesWithNeighbours[i][1]
        imgpairs[imgpairs_idx + 1] = imagesWithNeighbours[i][2]
        imgpairs_idx = imgpairs_idx + 2
    end
    
    return image.toDisplayTensor{input=imgpairs, nrow=#imagesWithNeighbours}
end

-- Selects N random images from a tensor of images.
-- @param tensor Tensor of images
-- @param n Number of random images to select
-- @returns List/table of images
function selectRandomImagesFrom(tensor, n)
    local shuffle = torch.randperm(tensor:size(1))
    local result = {}
    for i=1,math.min(n, tensor:size(1)) do
        table.insert(result, tensor[ shuffle[i] ])
    end
    return result
end

-- Loads all necessary models/networks and returns them.
-- Also loads normalization settings, which currently should always be nil.
-- @returns G, D, G_c2f22, D_c2f22, G_c2f32, D_c2f32, normMean, normStd
function loadModels()
    local file
    
    -- load G base
    file = torch.load(paths.concat(OPT.save_base, OPT.G_base))
    local G = file.G
    G:evaluate()
    
    -- load D base
    file = torch.load(paths.concat(OPT.save_base, OPT.D_base))
    local D = file.D
    D:evaluate()
    
    -- load G c2f 16 to 22
    file = torch.load(paths.concat(OPT.save_c2f22, OPT.G_c2f22))
    local G_c2f22 = file.G
    G_c2f22:evaluate()
    local normMean = file.normalize_mean
    local normStd = file.normalize_std
    
    -- load D c2f 16 to 22
    file = torch.load(paths.concat(OPT.save_c2f22, OPT.D_c2f22))
    local D_c2f22 = file.D
    D_c2f22:evaluate()
    
    -- load G c2f 22 to 32
    file = torch.load(paths.concat(OPT.save_c2f32, OPT.G_c2f32))
    local G_c2f32 = file.G
    G_c2f32:evaluate()
    
    -- load D c2f 22 to 32
    file = torch.load(paths.concat(OPT.save_c2f32, OPT.D_c2f32))
    local D_c2f32 = file.D
    D_c2f32:evaluate()
    
    return G, D, G_c2f22, D_c2f22, G_c2f32, D_c2f32, normMean, normStd
end

main()
