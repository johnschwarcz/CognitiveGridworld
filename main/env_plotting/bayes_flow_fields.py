import numpy as np; from env.env_helpers import Env_helpers

class Bayes_flow_fields(Env_helpers):

    def get_flow(self):
        self.joint_particle = self.get_particle_traj(self.joint_dist, naive = False)
        self.joint_flow, self.joint_flow_count = self.flow_loop(self.joint_particle)

        self.naive_particle = self.get_particle_traj(self.naive_dist, naive = True)
        self.naive_flow, self.naive_flow_count = self.flow_loop(self.naive_particle)

        self.agent_flow_counts = [self.joint_flow_count, self.naive_flow_count]
        self.agent_flows = [self.joint_flow, self.naive_flow]
        
    def get_particle_traj(self, dist, naive):                          # Only supported for 2 contexts
        inds = np.arange(self.realization_num**2)
        for t in self.step_range:
            if t == 0:
                cost = 1 
            else:
                cost = self.get_particle_dist(t, 0, naive)
                cost = cost + self.get_particle_dist(t, 1, naive)
            p = dist[:, t] / cost
            norm = self.avg_until(p, override = "sum", stop_shape = 2 if naive else 1)
            p /= norm
                   
            for b in self.batch_range:
                if naive:
                    i = np.random.choice(self.realization_range, p=p[b, 0])            
                    j = np.random.choice(self.realization_range, p=p[b, 1]) 
                else:
                    particle = np.random.choice(inds, p=p[b].ravel())            
                    i, j = np.unravel_index(particle, self.ctx_dims)
                    
                self.particle_traj[b,t, 0] = i
                self.particle_traj[b,t, 1] = j
        return self.particle_traj
            
    def get_particle_dist(self, t, c, naive):        
        particle = self.particle_traj[:, t-1,c, None]
        reals = self.realization_range[None, :]
        dist = (particle - reals)**2 + 1
        # dist = np.abs(particle - reals) + 1
        if c == 0 and not naive:
            return dist[:, :, None]
        return dist[:, None, :]
    
    def flow_loop(self, est, min_count = 100):
        # Get estimate flow field
        prev_est = np.roll(est, shift = 1, axis = 1)
        prev = np.digitize(prev_est, self.realization_range) - 1
        self.flow_count[:] = 0         
        self.flow_grid[:] = 0 
        diff = est - prev_est
        for x in self.realization_range:   
            x_bool = prev[..., 0] == x
            for y in self.realization_range:
                y_bool = prev[..., 1] == y
                for o in self.obs_range:
                    for i in range(2):
                        o_bool = self.obs_flat[..., o] == 1-i
                        full_bool = x_bool * y_bool * o_bool    
                        if full_bool.sum() > min_count:                             
                             self.flow_count[i, o, x, y] = full_bool.sum() 
                             self.flow_grid[i, o, x, y,:] = diff[full_bool].mean(0)
        return self.flow_grid.copy(), self.flow_count.copy()