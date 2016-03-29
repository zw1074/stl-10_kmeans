require 'nn'
require 'image'
require 'xlua'
--require 'iterm'
dofile './provider.lua'
--train = torch.load('train.t7')
provider = torch.load 'provider.t7'
test = torch.load('stl-10/test.t7b')
temp = torch.Tensor(4000, 3, 96, 96)
temp:copy(provider.trainData.data) 
temp2 = torch.Tensor(1000, 3, 96, 96)
temp2:copy(provider.valData.data)

test_data = {data = torch.Tensor(8000, 3, 96, 96), labels = torch.Tensor(8000)}
idx = 1
for i = 1, 10 do
   for j = 1, 800 do
      test_data.data[idx] = test.data[i][j]
      test_data.labels[idx] = i
      idx = idx + 1
   end
end

--Normalization
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1, 8000 do
   xlua.progress(i, 8000)
   local rgb = test_data.data[i]
   local yuv = image.rgb2yuv(rgb)
   yuv[1] = normalization(yuv[{{1}}])
   test_data.data[i] = yuv
end

for i = 2,3 do
   local mean = test_data.data:select(2,i):mean()
   local std = test_data.data:select(2,i):std()
   test_data.data:select(2,i):add(-mean)
   test_data.data:select(2,i):div(std)
end

function aug(data_in,n)
   indice = torch.randperm(data_in.data:size(1)):long():split(n)


   train_rotate = torch.Tensor(4000,3,96,96)
   for t, v in ipairs(indice) do
      rotate_para = torch.uniform(-0.3,0.3)
      scale_para = torch.uniform(100,120)
      translation_para_x = torch.uniform(0,5)
      translation_para_y = torch.uniform(0,5)
      xlua.progress(t,data_in.data:size(1)/(n))
      for i= 1,v:size(1) do
         data_in.data[v[i]] = image.crop( image.scale( image.rotate( image.translate(data_in.data[v[i]],translation_para_x, translation_para_y), rotate_para), scale_para),'c' ,96,96) 
         data_in.labels[v[i]] = data_in.labels[v[i]] 

      end
   end
   return data_in
end

a2 = torch.Tensor(4000, 3, 96, 96)
a2:copy(aug(provider.trainData, 100).data)
a3 = torch.Tensor(1000, 3, 96, 96)
a3:copy(aug(provider.valData, 100).data)
a4 = torch.Tensor(4000, 3, 96, 96)
a4:copy(aug(provider.trainData, 100).data)
a5 = torch.Tensor(1000, 3, 96, 96)
a5:copy(aug(provider.valData, 100).data)
a6 = torch.Tensor(4000, 3, 96, 96)
a6:copy(aug(provider.trainData, 100).data)
a7 = torch.Tensor(1000, 3, 96, 96)
a7:copy(aug(provider.valData, 100).data)


--a4 = torch.Tensor(4000, 3, 96, 96)
--a4:copy(aug(provider.trainData, 100).data)
--a5 = torch.Tensor(4000, 3, 96, 96)
--a5:copy(aug(provider.trainData, 100).data)

provider.trainData.data = torch.Tensor(20000,3,96,96)
provider.trainData.data = torch.cat({temp,temp2,a2, a3, a4, a5, a6, a7}, 1)
a2 = nil
a3 = nil
a4 = nil
a5 = nil
a6 = nil
a7 = nil
label = torch.cat({provider.trainData.labels, provider.valData.labels, provider.trainData.labels, provider.valData.labels, provider.trainData.labels, provider.valData.labels, provider.trainData.labels, provider.valData.labels},1)
provider.trainData.labels = label
provider.valData.data = torch.Tensor(8000, 3, 96, 96)
provider.valData.data = test_data.data
provider.valData.labels = test_data.labels
--provider.extraData.data = 0
torch.save('provider_full.t7', provider)
