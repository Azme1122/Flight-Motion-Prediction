from __future__ import print_function
from six.moves import cPickle as pickle
from utils.helper import generate_unique_randoms

import numpy as np
import random

try:
    
    from termcolor import colored
    
except ModuleNotFoundError:
    
    def colored(text, color=None):
        
        return text


class DataLoader:
    """3D trajectory dataset data loader class
    """
    
    def __init__(self, cfg=''):
        """Init and setup
        """
        
        # variables
        self.cfg = cfg
        self.name = cfg.name
        self.train_data_paths = cfg.paths['train_data_paths']
        self.eval_data_paths = cfg.paths['val_data_paths']
        self.test_data_paths = cfg.paths['test_data_paths']
        self.randomize = cfg.train_params['randomize_train_data']
        self.train_data_reduction = cfg.train_params['train_data_reduction']
        self.eval_data_reduction = cfg.train_params['eval_data_reduction']
        self.lstm_input_shape = cfg.model_params['lstm_input_shape']
        self.with_print = cfg.with_print
        
        # data storages
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        
        return
        
        
    def load_train_data(self):
        """Load training data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading train data: {self.train_data_paths}", 'green'))
        self.train_data = self.load_pkl_data(paths=self.train_data_paths, type='train')
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.train_data[0])} samples for training", 'green'))
        if self.with_print: print(colored(f" with randomized training data: {self.randomize}", 'green'))
        if self.with_print: print(colored(f" data reduction scale: {self.train_data_reduction}", 'green'))
        if self.with_print: print(colored(f" train X shape: {self.train_data[0].shape}, y shape: {self.train_data[1].shape}", 'green'))
        
        return
        
        
    def load_eval_data(self):
        """Load train-eval data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading eval data: {self.eval_data_paths}", 'green'))
        self.eval_data = self.load_pkl_data(paths=self.eval_data_paths, type='eval')
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.eval_data[0])} samples for train-eval", 'green'))
        if self.with_print: print(colored(f" data reduction scale: {self.eval_data_reduction}", 'green'))
        if self.with_print: print(colored(f" eval X shape: {self.eval_data[0].shape}, y shape: {self.eval_data[1].shape}", 'green'))
        
        return
        
        
    def load_test_data(self):
        """Load test data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading test data: {self.test_data_paths}", 'green'))
        self.test_data = self.load_pkl_data(paths=self.test_data_paths, type='test')
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.test_data[0])} samples for testing", 'green'))
        if self.with_print: print(colored(f" test X shape: {self.test_data[0].shape}, y shape: {self.test_data[1].shape}", 'green'))
        
        return
    
    
    def get_train_data(self):
        """Get training data for next epoch
        """
        
        # with random shuffle
        if self.randomize:
            
            shuffled_indices = list(range(0, len(self.train_data[0])))
            random.shuffle(shuffled_indices)
            self.train_data[0] = np.take(a=self.train_data[0], indices=shuffled_indices, axis=0)
            self.train_data[1] = np.take(a=self.train_data[1], indices=shuffled_indices, axis=0)
            
        # reduce train data to a random subset
        if self.train_data_reduction < 1.0:
            
            n = int(len(self.train_data[0]) * self.train_data_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.train_data[0])-1)
            X = np.take(a=self.train_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.train_data[1], indices=random_reduced_indices, axis=0)
            
            return X, y
        
        # with no modifications
        else:
            
            return self.train_data[0], self.train_data[1]
    
    
    def get_eval_data(self):
        """Get eval data for next epoch
        """
        
        # reduce train data to a random subset
        if self.eval_data_reduction < 1.0:
            
            n = int(len(self.eval_data[0]) * self.eval_data_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.eval_data[0])-1)
            X = np.take(a=self.eval_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.eval_data[1], indices=random_reduced_indices, axis=0)
            p = np.take(a=self.eval_data[2], indices=random_reduced_indices, axis=0)
            r = np.take(a=self.eval_data[3], indices=random_reduced_indices, axis=0)
            s = np.take(a=self.eval_data[4], indices=random_reduced_indices, axis=0)
            iw = np.take(a=self.eval_data[5], indices=random_reduced_indices, axis=0)
            tw = np.take(a=self.eval_data[6], indices=random_reduced_indices, axis=0)
            
            return X, y, p, r, s, iw, tw
        
        # with no modifications
        else:
            
            return self.eval_data[0], self.eval_data[1], self.eval_data[2], self.eval_data[3], self.eval_data[4], self.eval_data[5], self.eval_data[6]
        
    
    def get_test_data(self):
        """Get test data for next epoch
        """
        
        return self.test_data[0], self.test_data[1], self.test_data[2], self.test_data[3], self.test_data[4], self.test_data[5], self.test_data[6]
    
    
    def reorder_list(lst, indices):
        """Create a reordered copy of a given list
        """
        
        # create an empty list to store the elements in their new order
        tmp = []
        
        # iterate over the indices and add the corresponding elements from the original list to the new list
        for index in indices:
            
            if 0 <= index < len(lst):
                
                tmp.append(lst[index])
                
        return tmp
    
    
    def load_pkl_data(self, paths, type):
        """Load 3D track and transformation data from a .pkl file
        """
        
        nX = []
        ny = []
        npos = []
        nrot = []
        nsrc = []
        ninput_world = []
        ntarget_world = []
        
        # load data from source paths
        for p in paths:
        
            with open(p, 'rb') as f:
                sample_data = pickle.load(f)
                
            for id in sample_data:
                
                nX.append(sample_data[id]['X'])
                ny.append(sample_data[id]['y'][...,:3])
                npos.append(sample_data[id]['reference_position'])
                nrot.append(sample_data[id]['rotation_angle'])
                nsrc.append(sample_data[id]['source'])
                ninput_world.append(sample_data[id].get('input_world', np.zeros((sample_data[id]['X'].shape[0], 3))))
                ntarget_world.append(sample_data[id].get('target_world', np.zeros((sample_data[id]['y'].shape[0], 3))))
        
        X = np.array(nX)
        y = np.array(ny)
        pos = np.array(npos)
        rot = np.array(nrot)
        src = np.array(nsrc)
        input_world = np.array(ninput_world)
        target_world = np.array(ntarget_world)
        
        if X.ndim != 3 or X.shape[-1] != self.lstm_input_shape:
            
            raise ValueError(f"Expected X shape [N, obs_len, {self.lstm_input_shape}], got {X.shape}")
        
        if y.ndim != 3 or y.shape[-1] != 3:
            
            raise ValueError(f"Expected y shape [N, pred_len, 3], got {y.shape}")
        
        return [X, y, pos, rot, src, input_world, target_world]
