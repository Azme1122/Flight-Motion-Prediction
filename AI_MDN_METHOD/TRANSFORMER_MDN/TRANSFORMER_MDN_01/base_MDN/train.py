import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
import sys

from base_transformer import Transformer_Trajectory_Forecast, NLL_MDN_loss
from mdn import MDN_Trainer
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


def training(cfg, gpu_id):
    """Run training framework.
    """
    
    # start the training
    print(colored(f"Starting training: {cfg.name} on GPU: {gpu_id}", 'green'))
    
    # init dataloader
    data_loader = DataLoader(cfg=cfg)
    
    # load data
    data_loader.load_train_data()
    data_loader.load_eval_data()
    
    # logger
    train_logger = None
    log_file_handler = None
    
    if cfg.with_log:
        
        log_file_path = os.path.join(cfg.evaluation_path, 'training.log')
        os.remove(log_file_path) if os.path.exists(log_file_path) else None
        train_logger = logging.getLogger(f'training_{cfg.name}')
        train_logger.setLevel(logging.INFO)
        log_file_handler = logging.FileHandler(log_file_path)
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        train_logger.addHandler(log_file_handler)
    
    #--- init network
    # mixture density network
    # set cuda gpu device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    epoch = 1
    loss_hist = []
    model = Transformer_Trajectory_Forecast(cfg=cfg.model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.train_params['lr_default'])
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.train_params['lr_start_factor'],
        end_factor=cfg.train_params['lr_end_factor'],
        total_iters=cfg.train_params['train_epochs']
    )
    loss_fn = lambda output, target: NLL_MDN_loss(output=output, target=target, num_gaussians=cfg.model_params['num_gaussians'])
    
    # load pretrained model
    if cfg.train_params['resume_training']:
        
        # check if model exists and load it
        if os.path.exists(os.path.join(cfg.checkpoint_path, "model_final.pt")):
            
            checkpoint = torch.load(f=os.path.join(cfg.checkpoint_path, "model_final.pt"), map_location='cpu')
            model.load_state_dict(checkpoint["model"])
            loss_hist = checkpoint['loss']
            epoch = checkpoint['epoch']
            if cfg.with_print: print(colored(f"Resume training on gpu: {gpu_id}, load pretrained model \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - epoch: {epoch}", 'green'))
            if cfg.with_log: train_logger.info(f"Resume training on gpu: {gpu_id}, load pretrained model \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - epoch: {epoch}")
            
        # does not exist
        else:
            
            # stop
            if cfg.with_print: print(colored(f"Error: Try to resume training, but pretrained model does not exist, please check path: {os.path.join(cfg.checkpoint_path, 'model_final.pt')}", 'red'))
            if cfg.with_log: train_logger.info(f"Error: Try to resume training, but pretrained model does not exist, please check path: {os.path.join(cfg.checkpoint_path, 'model_final.pt')}")
            if cfg.with_print: print(colored(f"Training finished...", 'red'))
            if cfg.with_log: train_logger.info(f"Training finished...")
            return -1
        
    # train from scratch
    else:
        
        if cfg.with_print: print(colored(f"Start training from scratch for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - gpu: {gpu_id}", 'green'))
        if cfg.with_log: train_logger.info(f"Start training from scratch for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - type: {cfg.type} \n - gpu: {gpu_id}")
        
    #--- start training
    # build trainer
    trainer = MDN_Trainer(cfg=cfg, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch, loss_hist=loss_hist, logger=train_logger)
    final_epoch, _ = trainer.train(data_loader=data_loader)
    
    # Save final model
    trainer.save(epoch=final_epoch, diverged=False, final=True)
        
    if cfg.with_print: print(colored(f"Saved final model with {count_model_parameters(model=model)} parameters", 'green'))
    if cfg.with_log: train_logger.info(f"Saved final model with {count_model_parameters(model=model)} parameters")
    if cfg.with_print: print(colored(f"All trainings completed, shutdown...", 'green'))
    if cfg.with_log: train_logger.info(f"All trainings completed, shutdown...")
    
    if cfg.with_log:
        
        log_file_handler.close()
        train_logger.removeHandler(log_file_handler)
        
    print(colored(f"Finished training: {cfg.name} on GPU: {gpu_id}", 'cyan'))
    return
    
    
if __name__ == "__main__":
    
    # parse arguments
    model_arch = 'base_mdn'
    type = 'training'
    parser = config_parser()
    args = parser.parse_args()
    
    # gpu handling
    if args.gpu:
        
        gpu_id = args.gpu
            
    else:
        
        gpu_id = 0
    
    # load configs from file
    config_dir = get_config_dir(model_arch=model_arch, target=args.target)
    
    # multiple configs
    if args.configs == 'all':
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, conf), target=args.target, with_log=args.log, with_print=args.print, name=conf[:-5], model_arch=model_arch, type=type) for conf in os.listdir(config_dir) if conf.endswith('.json')]
    
    # single config, selected by user
    else:
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, args.configs), target=args.target, with_log=args.log, with_print=args.print, name=args.configs[:-5], model_arch=model_arch, type=type)]
    
    for cfg in configs:
    
        training(cfg=cfg, gpu_id=gpu_id)
        
    sys.exit()
