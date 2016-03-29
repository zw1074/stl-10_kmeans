-- input is 3X96X96 image, output is a 3X32X32 patch selected based on the gradient

function find_patch(picture, input_size, output_size, num_channel)
	local vertical_diff = picture[{{},{2, input_size},{}}] - picture[{{}, {1, input_size - 1},{}}] --vertical differential
	local horizontal_diff = picture[{{},{},{2, input_size}}] - picture[{{}, {},{1, input_size - 1}}] --horizontal differential
	local power2_matrix = torch.Tensor(num_channel, input_size - 1, input_size - 1):fill(2)
	local gradient = vertical_diff[{{},{},{1,input_size - 1}}]:cpow(power2_matrix) + horizontal_diff[{{},{1,input_size - 1},{}}]:cpow(power2_matrix) --gradient norm
	local gradient_max = torch.max(gradient, 1) --max gradient norm in 3 channels 1X95X95

	-- summation of gradient norm in every 32X32 patches
	local gradient_sum = torch.Tensor(1, input_size - output_size, input_size - output_size):fill(0)  --1X64X64
	for i = 1, input_size - output_size do
    	for j =1, input_size - output_size do
        gradient_sum[1][i][j] = torch.sum(gradient_max[{{}, {i, i+2}, {j, j+2}}])
    	end
	end

	local gradient_flatten = gradient_sum:resize(1, (input_size - output_size)*(input_size - output_size)) --1X4096
	
	-- find top 10 
	y, index = torch.topk(gradient_flatten, 10, true)
	-- randon integer between [1, 10]
	choice = torch.random(1,10)
    choice_2 = torch.random(1,10)
	-- random choice 
	loc_in_flat = index[1][choice]
    loc_in_flat_2 = index[1][choice_2]
	i = 1
	while loc_in_flat > input_size - output_size do
    	i = i + 1
    	loc_in_flat = loc_in_flat - (input_size - output_size)
	end
	j = loc_in_flat
    i_2 = 1
	while loc_in_flat_2 > input_size - output_size do
    	i_2 = i_2 + 1
    	loc_in_flat_2 = loc_in_flat_2 - (input_size - output_size)
	end
	j_2 = loc_in_flat_2
    
    local patch = torch.Tensor(2,num_channel, output_size, output_size)
	patch[1] = picture[{{},{i, i+2},{j, j+2}}]:float()
    patch[2] = picture[{{}, {i_2, i_2 + 2}, {j_2, j_2 + 2}}]:float()
	--local index_x = j + 15
	--local index_y = i + 15 
	return patch-- index_x, index_y
end
