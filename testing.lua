require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

print("loading testing data...")
t  = torch.load('provider_full.t7')
test.data = t.valData.data
test.label = t.valData.labels
t = nil

print("loading model...")
model = torch.load('logs/model.net')


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

