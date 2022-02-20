import numpy as np

# helper function
def getInfo(D, alg_num, col_name):
	return D[algo_list[alg_num]].item().get(col_name)

# initialize constant variables
algo_list = ["RGB"] # <--- will add more algorithms in the future
cols_list = ["rewards", "reward", "value", "chosen_action", "binary_action", "prob_actions"]
n_algos = len(algo_list)
n_cameras = 3
n_steps = 1000	# number of time steps in the motion sequence

# load data from npy files
DATA = {}
DATA[algo_list[0]] = np.load(algo_list[0]+'_results.npy', allow_pickle=True, encoding="latin1") 
action_mappings = np.load("action_mappings.npy",allow_pickle = True)
action_mappings = action_mappings.item()

# extract chosen actions
chosen_actions = np.zeros((n_algos, n_steps, 2*n_cameras))
value_list = list(action_mappings.values())	

for i in range(n_algos):
	action_keys = getInfo(DATA, i, cols_list[3])[0,:]
	print("algorithm: "+algo_list[i])

	for j in range(n_steps):
		a = int(action_keys[j])
		chosen_actions[i,j,:] = value_list[a]-1
		print("\tstep "+str(j)+": action "+str(a)+"\t= "+str(chosen_actions[i,j,:]))