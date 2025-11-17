from .bayes_likelihood import Bayes_likelihood
import numpy as np

class Env_Customization(Bayes_likelihood):

    def EC_gen_state_embeddings(self):
        if not self.custom_state_embeddings():
            self.default_state_embeddings()
            
    def EC_gen_context(self):
        if not self.custom_context():
            self.default_context()

    def EC_gen_likelihoods(self):
        if not self.custom_likelihoods():
            self.default_likelihoods()

    def EC_gen_realizations(self):
        if not self.custom_realizations():
            self.default_realizations()

    def EC_gen_observations(self):
        if not self.custom_gen_observations():
            self.default_gen_observations()
        
    ########################################################################################################
    """ EC_gen_state_embeddings: state embeddings """ 
    ########################################################################################################
    def custom_state_embeddings(self, using_custom = False):
        """ 
        all_K, all_Q : vectors representing the state embeddings.
        V_range      : vector representing the shared value vector.
        roll_V_range : possible rollings of V_range, needed by the custom likelihood.
        """
        return using_custom

    ########################################################################################################
    """ EC_gen_context: goal & context selection """ 
    ########################################################################################################
    def custom_context(self, using_custom = False):
        """
        ctx_inds : indices of active states.
        goal_ind : indices of the reward-dependent state.
        ctx_turnover : probability of a new context.
        """
        return using_custom        

    ########################################################################################################
    """ EC_gen_likelihoods: joint & marginalized likelihood """ 
    ########################################################################################################
    def custom_likelihoods(self, using_custom = False):
        """
        joint_Z          : scalars representing the interactions between active states.
        joint_likelihood : a tensor of all possible joint realizations (for the entire batch & all observations).
        naive_likelihood : marginalization over the joint likelihood to remove the mutual information.
        """
        return using_custom

    ########################################################################################################
    """ EC_gen_realizations: sample context values """ 
    ########################################################################################################
    def custom_realizations(self, using_custom = False):
        """
        ctx_vals    : scalars representing sampled realizations for each active state.
        goal_value  : the value of the reward-dependent state.
        pobs__joint : scalaras representing the probability of each observation given the joint realizations.
        """
        return using_custom
    
    ########################################################################################################
    """ EC_gen_observations: sample trajectory of observations """ 
    ########################################################################################################
    
    def custom_gen_observations(self, using_custom = False):
        """
        obs_flat : a tensor of observations for each batch and step.
        """
        return using_custom