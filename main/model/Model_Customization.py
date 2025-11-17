import torch; import torch.nn as nn; 

class Model_Customization(nn.Module):
    def __init__(self):
        super(Model_Customization, self).__init__()       

    def MC_init_internal_embeddings(self):
        if not self.custom_internal_embeddings():
            self.default_internal_embeddings()

    def MC_init_classifier(self):
        if not self.custom_classifier():
            self.default_classifier()

    def MC_init_generator(self):
        if not self.custom_generator():
            self.default_generator()

    def MC_to_interactions(self):
        if not self.custom_interactions():
            self.default_interactions()

    def MC_to_classification(self):
        if not self.custom_classification():
            self.default_classification()

    def MC_to_pobs(self):
        if not self.custom_pobs():
            self.default_pobs()

    ########################################################################################################
    def custom_internal_embeddings(self, using_custom = False):
        """ 
        all_K, all_Q : the classifiers' internal estimate of the state embeddings, Defaults are ground truth).
        """ 
        return using_custom        
    ########################################################################################################
    def custom_classifier(self, using_custom = False):
        """ 
        classifier_optim : the optimizer for the classifier network.
        """         
        return using_custom
    ########################################################################################################
    def custom_generator(self, using_custom = False):
        self.generator_optim = None
        """
        generator_optim : the optimizer for the generator network.
        """
        return using_custom      
    ########################################################################################################
    def custom_interactions(self, using_custom_interactions = False):
        """
        active_K, active_Q, active_Z : the active K, Q, Z estimates
        """
        return using_custom_interactions       
    ########################################################################################################
    def custom_classification(self, using_custom_classification = False):        
        """
        classifier_belief_flat: the marginal belief over the all states
        classifier_goal_belief: the marginal belief over the goal state
        classifier_goal_selection: sample from classifier_goal_belief
        """
        return using_custom_classification   
    #######################################################################################################
    def custom_pobs(self, using_custom_pobs = False):
        """
        pred_pobs: predicted P(obs | ...)
        """
        return using_custom_pobs   

    #######################################################################################################