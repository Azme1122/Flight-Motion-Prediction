import os
import torch
import logging

from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from generate_eval_summary import generate_summary_table
from utils.config_loader import ConfigLoader
from utils.helper import config_parser, count_model_parameters
from utils.data_loader import DataLoader

try:
    
    from termcolor import colored
    
except ModuleNotFoundError:
    
    def colored(text, color=None):
        
        return text


def get_config_dir(model_arch, target):
    """Resolve config directory for local or src-style project layouts.
    """
    
    local_config_dir = os.path.join(os.getcwd(), 'configs', target)
    src_config_dir = os.path.join(os.getcwd(), 'src', model_arch, 'configs', target)
    
    if os.path.exists(local_config_dir):
        
        return local_config_dir
    
    return src_config_dir


def testing(args, gpu_id):
    """Run testing framework.
    """
    
    #--- init and setup
    model_arch = 'base_mdn'
    type = 'testing'
    
    print(colored(f"Starting testing: on GPU: {gpu_id}", 'green'))
    
    # load configs from file
    config_dir = get_config_dir(model_arch=model_arch, target=args.target)
    
    # multiple configs
    if args.configs == 'all':
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, conf), target=args.target, with_log=args.log, with_print=args.print, name=conf[:-5], model_arch=model_arch, type=type) for conf in os.listdir(config_dir) if conf.endswith('.json')]
    
    # single config, selected by user
    else:
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, args.configs), target=args.target, with_log=args.log, with_print=args.print, name=args.configs[:-5], model_arch=model_arch, type=type)]
    
    # start the testings
    for cfg in configs:
        
        # set cuda gpu device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init dataloader
        data_loader = DataLoader(cfg=cfg)
        
        # load data
        data_loader.load_test_data()
        
        # logger
        test_logger = None
        log_file_handler = None
        
        if cfg.with_log:
            
            log_file_path = os.path.join(cfg.testing_path, 'testing.log')
            os.remove(log_file_path) if os.path.exists(log_file_path) else None
            test_logger = logging.getLogger(f'testing_{cfg.name}')
            test_logger.setLevel(logging.INFO)
            log_file_handler = logging.FileHandler(log_file_path)
            log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            test_logger.addHandler(log_file_handler)
        
        # load a trained model
        model = LSTM_Trajectory_Forecast(cfg=cfg.model_params)
        model_path = os.path.join(cfg.checkpoint_path, "model_final.pt")
        
        if not os.path.exists(model_path):
            
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        
        model.load_state_dict(torch.load(f=model_path, map_location=device)["model"])
        
        # create eval forecaster
        eval_forecaster = MDN_Forecaster(cfg=cfg, model=model, data_loader=data_loader, type='testing', device=device, logger=test_logger)
        
        if cfg.with_print: print(colored(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - model parameters: {count_model_parameters(model=model)}", 'green'))
        if cfg.with_log: test_logger.info(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - model parameters: {count_model_parameters(model=model)}")
        
        # Run evaluation tasks
        eval_forecaster.evaluate(epoch=None)
        summary_path = generate_summary_table(
            testing_dir=cfg.testing_path,
            model_name="LSTM-MDN",
            title="LSTM-MDN Local-Step Evaluation Summary"
        )
        if cfg.with_print: print(colored(f"Saved evaluation summary: {summary_path}", 'magenta'))
        if cfg.with_log: test_logger.info(f"Saved evaluation summary: {summary_path}")
        
        # save example plots
        if cfg.test_params['plot_examples'] or cfg.test_params['plot_examples_to_map']:
            
            if cfg.with_print: print(colored(f"Plotting {int(len(data_loader.test_data[0])/cfg.test_params['plot_step'])} examples...", 'magenta'))
            if cfg.with_log: test_logger.info(f"Plotting {int(len(data_loader.test_data[0])/cfg.test_params['plot_step'])} examples...")
            
            eval_forecaster.save_examples(
                epoch=None, 
                n_samples=cfg.test_params['num_samples'],
                mesh_range_x=cfg.test_params['mesh_range_x'],
                mesh_range_y=cfg.test_params['mesh_range_y'], 
                mesh_resolution=cfg.test_params['mesh_resolution'], 
                confidence_levels=cfg.test_params['confidence_levels'],
                plot_local=cfg.test_params['plot_examples'],
                plot_map=cfg.test_params['plot_examples_to_map']
            )
        
        if cfg.with_print: print(colored(f"Finished testing...", 'green'))
        if cfg.with_log: test_logger.info(f"Finished testing...")
    
    if cfg.with_print: print(colored(f"All tests completed, shutdown...", 'green'))
    if cfg.with_log: test_logger.info(f"All tests completed, shutdown...")
    
    if cfg.with_log:
        
        log_file_handler.close()
        test_logger.removeHandler(log_file_handler)
        
    print(colored(f"Finished testing: {cfg.name} on GPU: {gpu_id}", 'cyan'))
        
    return


if __name__ == "__main__":
    
    parser = config_parser()
    args = parser.parse_args()
    
    # gpu handling
    if args.gpu:
        
        gpu_id = args.gpu
        
    else:
        
        gpu_id = 0
        
    testing(args=args, gpu_id=gpu_id)
