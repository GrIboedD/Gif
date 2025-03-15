from global_optimization import global_optimization
from visualization import visualize_k_loss, visualize_loss_function
from two_step_optimization import two_step_optimization
import optimization_2_param, optimization_4_param

#visualize_k_loss(323.15, 0.0310, [0, 7000], [7300, 9000], 1000)
#visualize_k_loss(323.15, 0.0310, [0, 15000], [0, 15000], 10000)
#visualize_loss_function([0, 0.1], [0, 0.1], 100)
#optimization_2_param.optimize()
#optimization_4_param.optimize()
global_optimization(0, 20000, 4)
#two_step_optimization()