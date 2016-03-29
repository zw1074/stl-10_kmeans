require 'xlua'
require 'optim'
require 'cunn'
require 'nn'
require 'image'
local c = require 'trepl.colorize'

print("Downloading testing data...")
if io.open('test.t7b') == nil then
   os.execute('wget https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b')
end
test = {data = torch.Tensor(8000, 3, 96, 96), label = torch.Tensor(8000)}
raw_data = torch.load('test.t7b')
idx = 1
print("Parse data")
for i = 1, 10 do
   for j = 1, 800 do
      test.data[idx] = raw_data.data[i][j]
      test.label[idx] = i
      idx = idx + 1
   end
end
raw_data = nil
--Normalization
print('normalization')
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1, 8000 do
   xlua.progress(i, 8000)
   local rgb = test.data[i]
   local yuv = image.rgb2yuv(rgb)
   yuv[1] = normalization(yuv[{{1}}])
   test.data[i] = yuv
end

for i = 2,3 do
   local mean = test.data:select(2,i):mean()
   local std = test.data:select(2,i):std()
   test.data:select(2,i):add(-mean)
   test.data:select(2,i):div(std)
end

print("Downloading model...")
if io.open('model.net') == nil then
   os.execute('wget http://www.cs.nyu.edu/~zw1074/model.net')
end
model = torch.load('model.net')


function testing_t()
  local confusion = optim.ConfusionMatrix(10)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  res = torch.Tensor((#test.data)[1],10):float():cuda()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,(#test.data)[1],bs do
    local outputs = model:forward(test.data:narrow(1,i,bs):cuda())
    confusion:batchAdd(outputs, test.label:narrow(1,i,bs):cuda())
    res[{{i,i+bs-1},{}}] = outputs
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
	return res
end

function prediction(data)
res = {}
for i = 1, (#data)[1] do
pred = data[i]:float()
res[i] = torch.nonzero(pred:eq(pred:max()))[1][1]  
end
return res
end

res = prediction(testing_t())
print("generating predictions ...")
f = io.open('predictions.csv', 'w')
f:write('Id'.. ','.. 'Prediction\n' )

for i = 1, #res do
   f:write(i ..','..res[i]..'\n')

end
f:close()

