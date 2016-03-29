require 'xlua'
require 'nn'
require 'unsup'
require 'image'
dofile('find_patch.lua')
--print('==> loading centroid')
--center = torch.load('centroid.t7b')
print('==> loading data')
raw_data = torch.load('stl-10/extra.t7b')

local MaxPooling = nn.SpatialMaxPooling

model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 3,3,1,1,1,1))
--model:get(1).weight = center:double()
model:add(nn.SpatialBatchNormalization(64,1e-3))
model:add(nn.ReLU(true))
model:add(MaxPooling(4,4,4,4):ceil())
--center = nil
data = torch.Tensor(10000, 3, 96, 96)
indexx = torch.randperm(100000)
for i = 1,10000 do
    xlua.progress(i, 10000)
    data[i] = raw_data.data[1][indexx[i]]:double()

end
raw_data = nil
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,10000 do
    xlua.progress(i, 10000)
    local yuv = image.rgb2yuv(data[i])
    yuv[1] = normalization(yuv[{{1}}])
    data[i] = yuv
end
normalization = nil
for i =2,3 do
    local mean = data:select(2,i):mean()
    local std = data:select(2,i):std()
    data:select(2,i):add(-mean)
    data:select(2,i):div(std)
end

print('==> Getting batch')
batch = torch.Tensor(20000, 3*3*3)
idx = 1
for i = 1,10000 do
    xlua.progress(i,10000)
    local patch = find_patch(data[i], 96, 3, 3)
    for j = 1,2 do
        batch[idx] = patch[j]
        idx = idx + 1
    end
end
centroid = unsup.kmeans(batch, 64, 1000)
torch.save('layer1', centroid)
model:get(1).weight = centroid

print(data:size())
bigdata = torch.Tensor(10000, 64, 24, 24)

for i = 1,10000 do
    xlua.progress(i, 10000)
    bigdata[{{i}, {}, {}, {}}] = model:forward(data[{{i}, {}, {}, {}}])
end

data = nil
--print('==> Normalization')

-- normalize every channel globally:
--for i = 1, 64 do
--    local mean_u = bigdata:select(2,i):mean()
--    local std_u = bigdata:select(2,i):std()
--    bigdata:select(2,i):add(-mean_u)
--    bigdata:select(2,i):div(std_u)
--end

print('==> Getting batch')
batch = torch.Tensor(20000, 64*3*3)
idx = 1
for i = 1,10000 do
    xlua.progress(i,10000)
    local patch = find_patch(bigdata[i], 24, 3, 64)
    for j = 1,2 do
        batch[idx] = patch[j]
        idx = idx + 1
    end
end
centroid = unsup.kmeans(batch, 64, 1000)
torch.save('layer2', centroid)

model = nn.Sequential()
model:add(nn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1))
model:get(1).weight = centroid
model:add(nn.SpatialBatchNormalization(64, 1e-3))
model:add(nn.ReLU(true))
--model:add(MaxPooling(3,3,3,3):ceil())
bigdata_2 = torch.Tensor(10000, 64, 24, 24)
for i = 1,10000 do
    xlua.progress(i, 10000)
    bigdata_2[{{i}, {}, {}, {}}] = model:forward(bigdata[{{i}, {}, {}, {}}])
end
bigdata = nil
data = nil
print('==> Normalization')

-- normalize every channel globally:
--for i = 1, 64 do
--    local mean_u = bigdata_2:select(2,i):mean()
--    local std_u = bigdata_2:select(2,i):std()
--    bigdata_2:select(2,i):add(-mean_u)
--    bigdata_2:select(2,i):div(std_u)
--end

print('==> Getting batch')
batch = torch.Tensor(20000, 64*3*3)
idx = 1
for i = 1,10000 do
    xlua.progress(i,10000)
    local patch = find_patch(bigdata_2[i], 24, 3, 64)
    for j = 1,2 do
        batch[idx] = patch[j]
        idx = idx + 1
    end
end
centroid = unsup.kmeans(batch, 128, 1000)
torch.save('layer3', centroid)

model = nn.Sequential()
model:add(nn.SpatialConvolution(64, 128, 3,3, 1,1, 1,1))
model:get(1).weight = centroid
model:add(nn.SpatialBatchNormalization(128, 1e-3))
model:add(nn.ReLU(true))
model:add(MaxPooling(3,3,3,3):ceil())
bigdata = torch.Tensor(10000, 128, 8, 8)
for i = 1,10000 do
    xlua.progress(i, 10000)
    bigdata[{{i}, {}, {}, {}}] = model:forward(bigdata_2[{{i}, {}, {}, {}}])
end
--bigdata = nil
--data = nil
--print('==> Normalization')

-- normalize every channel globally:
--for i = 1, 128 do
--    local mean_u = bigdata_2:select(2,i):mean()
--    local std_u = bigdata_2:select(2,i):std()
--    bigdata_2:select(2,i):add(-mean_u)
--    bigdata_2:select(2,i):div(std_u)
--end

print('==> Getting batch')
batch = torch.Tensor(20000, 128*3*3)
idx = 1
for i = 1,10000 do
    xlua.progress(i,10000)
    local patch = find_patch(bigdata[i], 8, 3, 128)
    for j = 1,2 do
        batch[idx] = patch[j]
        idx = idx + 1
    end
end
centroid = unsup.kmeans(batch, 256, 1000)
torch.save('layer4', centroid)

