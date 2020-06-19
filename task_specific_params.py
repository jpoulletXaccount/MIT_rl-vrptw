from collections import namedtuple

# task specific params
TaskVRP = namedtuple('TaskVRP', ['task_name',
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',

						'demand_max'])

TaskVRPTW = namedtuple('TaskVRPTW', ['task_name',
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',
						'demand_max'])


task_lst = {}

# VRP10
vrp10 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=11,
			  n_cust = 10,
			  decode_len=20,
			  capacity=20,
			  demand_max=9)
task_lst['vrp10'] = vrp10

# VRP10 transfer learning
vrp10_tl = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=11,
			  n_cust = 10,
			  decode_len=20,
			  capacity=30,
			  demand_max=9)
task_lst['vrp10_rl'] = vrp10_tl

# VRP20
vrp20 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=21,
			  n_cust = 20,
			  decode_len=30,
			  capacity=30,
			  demand_max=9)
task_lst['vrp20'] = vrp20

# VRP20 transfer learning
vrp20_tl = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=21,
			  n_cust = 20,
			  decode_len=45,
			  capacity=30,
			  demand_max=9)
task_lst['vrp20_rl'] = vrp20_tl


# VRP50
vrp50 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=51,
			  n_cust = 50,
			  decode_len=70,
			  capacity=40,
			  demand_max=9)
task_lst['vrp50'] = vrp50

# VRP100
vrp100 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=101,
			  n_cust = 100,
			  decode_len=140,
			  capacity=50,
			  demand_max=9)
task_lst['vrp100'] = vrp100

################################

# VRPTW10
vrptw10 = TaskVRPTW(task_name = 'vrptw',
			  input_dim=5,
			  n_nodes=11,
			  n_cust = 10,
			  decode_len=20,
			  capacity=20,
			  demand_max=9)
task_lst['vrptw10'] = vrptw10

# VRPTW20
vrptw20 = TaskVRPTW(task_name = 'vrptw',
			  input_dim=5,
			  n_nodes=21,
			  n_cust = 20,
			  decode_len=40,
			  capacity=25,
			  demand_max=9)
task_lst['vrptw20'] = vrptw20

# VRPTW50
vrptw50 = TaskVRPTW(task_name = 'vrptw',
			  input_dim=5,
			  n_nodes=51,
			  n_cust = 50,
			  decode_len=90,
			  capacity=40,
			  demand_max=9)
task_lst['vrptw50'] = vrptw50
