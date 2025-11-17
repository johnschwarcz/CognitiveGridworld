import numpy as np; from matplotlib.colors import LinearSegmentedColormap
from .env_generators import Env_generators

class Env_preprocessing(Env_generators):

    def preprocess_env(self):
        self.pre_ctx_dims = 2 
        self.test_set = True # default true 

        self.V_num = 1 + 2 * self.realization_num # enables probabilities to go 'out of bounds' of the likelihood during roll
        self.Z_num = max(1, self.ctx_num * (self.ctx_num-1))
        self.ctx_to_the_state = self.ctx_num ** self.state_num
        self.R_to_the_ctx = self.realization_num ** self.ctx_num
        self.ctx_realization_num = self.ctx_num * self.realization_num

        # Pre-allocate arrays and dimensions for various parameters.
        self.ctx_range = np.arange(self.ctx_num)
        self.obs_range = np.arange(self.obs_num)
        self.step_range = np.arange(self.step_num)
        self.state_range = np.arange(self.state_num)
        self.batch_range = np.arange(self.batch_num)
        self.roll_realization_range = np.arange(self.V_num)
        self.realization_range = np.arange(self.realization_num)          
        self.state_to_the_ctx_range = self.state_num ** self.ctx_range
        self.R_to_the_ctx_range = self.realization_num ** self.ctx_range

        V_range = self.likelihood_temp * np.sin(np.linspace(-1, 1, self.V_num + 1)[1:] * np.pi)
        self.roll_V_range = np.zeros((self.realization_num, self.V_num))
        for r in range(self.realization_num):
            self.roll_V_range[r, :] = np.roll(V_range, r)

        # Define useful shapes for array generation.
        self.state_obs_dims = (self.state_num, self.obs_num)
        self.batch_ctx_dims = (self.batch_num, self.ctx_num)
        self.batch_obs_dims = (self.batch_num, self.obs_num)
        self.batch_step_dims = (self.batch_num, self.step_num)
        self.ctx_dims = tuple([self.realization_num] * self.ctx_num)
        self.ctx_realization_dims = (self.ctx_num, self.realization_num)
        self.BSCR_dims = (*self.batch_step_dims, *self.ctx_realization_dims)
        self.Z_dims = (*self.batch_obs_dims, self.Z_num)
        self.V_dims = (*self.Z_dims, self.realization_num)        
        self.L_dims = (*self.batch_obs_dims, *self.ctx_dims)
        
        # Pre-allocate arrays for future use.
        self.model_perf_log = np.zeros((self.checkpoint_every, self.step_num, 3))   # [acc, TP, mse]
        self.particle_traj = np.zeros((*self.batch_step_dims, self.ctx_num))
        self.flow_count = np.zeros((2, self.obs_num, self.realization_num, self.realization_num))
        self.flow_grid = np.zeros((2, self.obs_num, self.realization_num, self.realization_num, 2))
        self.ctx_inds, self.ctx_vals = [np.zeros(self.batch_ctx_dims, dtype=int) for _ in range(2)]

        # Preprocess plotting, flow.
        self.joint_col = "C0"
        self.goal_col = '#78c765'
        self.naive_col = "#A9A9A9"
        self.agent_titles = ["joint", "naive"]
        self.agent_cols = [self.joint_col, self.naive_col]
        self.joint_map = LinearSegmentedColormap.from_list("c", [(1, 1, 1, 0), self.joint_col])
        self.naive_map = LinearSegmentedColormap.from_list("c", [(1, 1, 1, 0), self.naive_col])
        self.agent_col_maps = [self.joint_map, self.naive_map]
