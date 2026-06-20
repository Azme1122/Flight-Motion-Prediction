import os
import json
import copy
import numpy as np

try:
    
    from termcolor import colored
    
except ModuleNotFoundError:
    
    def colored(text, color=None):
        
        return text

try:
    
    import seaborn as sns
    
except ModuleNotFoundError:
    
    sns = None


class ConfigLoader:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, config_path, target, with_log, with_print, name, model_arch, type):
        """ Init and setup 
        """
        
        # load config from file
        if with_print: print(colored(f"Loading config data for config: {name}", 'green'))
        
        with open(config_path) as f:
            cfg = json.load(f)
        
        # variables and paths
        self.name = name
        self.target = target
        self.model_arch = model_arch
        self.type = type
        self.with_log = with_log
        self.with_print = with_print
        self.paths = cfg['paths']
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']
        self.test_params = cfg['test_params']
        self.eval_metrics = cfg['eval_metrics']
        
        # eval and plotting
        self.reliability_bins = [k for k in np.arange(0.0, 1.01, 0.01)]
        if sns:
            
            palette = np.array(sns.color_palette(palette='colorblind', n_colors=len(self.test_params['test_horizons'])))
            
        else:
            
            palette = np.array([
                [0.00, 0.45, 0.70],
                [0.87, 0.56, 0.02],
                [0.00, 0.62, 0.45],
                [0.84, 0.37, 0.00],
                [0.80, 0.47, 0.65],
                [0.34, 0.71, 0.91]
            ][:len(self.test_params['test_horizons'])])
            
        self.colors_rgb = copy.deepcopy(palette)
        self.colors_bgr = copy.deepcopy(palette)
        self.colors_bgr[:, [2, 0]] = self.colors_bgr[:, [0, 2]]
        
        # paths
        self.result_path = os.path.join(self.paths['result_path'], self.model_arch, target, self.name)
        self.checkpoint_path = os.path.join(self.result_path, 'checkpoints')
        self.evaluation_path = os.path.join(self.result_path, 'evaluation')
        self.eval_ego_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'ego')
        self.eval_world_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'world')
        self.testing_path = os.path.join(self.result_path, 'testing')
        self.test_ego_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'ego')
        self.test_world_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'world')
        
        # create result directory structure
        if not os.path.exists(self.result_path): os.makedirs(self.result_path) 
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.evaluation_path): os.makedirs(self.evaluation_path)
        if not os.path.exists(self.eval_ego_examples_path): os.makedirs(self.eval_ego_examples_path)
        if not os.path.exists(self.eval_world_examples_path): os.makedirs(self.eval_world_examples_path)
        if not os.path.exists(self.testing_path): os.makedirs(self.testing_path)
        if not os.path.exists(self.test_ego_examples_path): os.makedirs(self.test_ego_examples_path)
        if not os.path.exists(self.test_world_examples_path): os.makedirs(self.test_world_examples_path)
