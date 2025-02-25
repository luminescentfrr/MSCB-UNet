import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.mscbunet import MSCB_UNet

from engine import *
import os
import sys
import time
import warnings
from utils import *
from configs.config_setting import *

warnings.filterwarnings("ignore")

def ensure_dir(directory):
    """Ensure the directory exists, create it if not."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")
            raise

def check_write_permissions(directory):
    """Check write permissions for the directory."""
    try:
        test_file = os.path.join(directory, 'test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"No write permission for directory {directory}: {str(e)}")
        return False

def check_and_fix_directory(config):
    """Check and fix the working directory configuration."""
    if os.path.exists('/root/autodl-tmp'):  # AutoDL environment
        # If base_dir points to a read-only file system
        if config.base_dir.startswith('/autodl-fs'):
            # Change to a writable location
            new_base_dir = config.base_dir.replace('/autodl-fs/data', '/root/autodl-tmp')
            print(f"Redirecting base_dir from {config.base_dir} to {new_base_dir}")
            config.base_dir = new_base_dir
            config.work_dir = os.path.join(config.base_dir, 
                                         os.path.basename(config.work_dir))
    
    # Ensure the directory exists
    os.makedirs(config.base_dir, exist_ok=True)
    os.makedirs(config.work_dir, exist_ok=True)
    
    # Test directory write access
    test_file = os.path.join(config.work_dir, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Successfully verified write access to {config.work_dir}")
        return True
    except Exception as e:
        print(f"Error: Cannot write to directory {config.work_dir}")
        print(f"Error details: {str(e)}")
        return False

def main(config):
    """
    Main training function
    Args:
        config: Configuration object
    """
    print('#----------Checking working directory----------#')
    if not check_and_fix_directory(config):
        print("Error: Could not set up writable working directory")
        print("Please check your permissions or use a different location")
        sys.exit(1)
    
    print('#----------Creating working directory----------#')
    # Create necessary directories
    sys.path.append(config.work_dir + '/')
    ensure_dir(config.work_dir)
    ensure_dir(os.path.join(config.work_dir, 'log'))
    ensure_dir(os.path.join(config.work_dir, 'checkpoints'))
    ensure_dir(os.path.join(config.work_dir, 'outputs'))
    
    resume_model = os.path.join(config.work_dir, 'checkpoints', 'latest.pth')
    
    # Create logger and TensorBoard writer
    global logger
    logger = get_logger('train', os.path.join(config.work_dir, 'log'))
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')
     
    # Log configuration information
    log_config_info(config, logger)
    
    print('#----------GPU Initialization----------#')
    # Set GPU and random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    
    print('#----------Preparing Dataset----------#')
    # Create training dataset loader
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        drop_last=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    # Create validation dataset loader
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        drop_last=False,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    print('#----------Creating Model----------#')
    # Create model
    model = MSCB_UNet(
        in_channels=config.input_channels,
        num_classes=config.num_classes,
        base_c=config.model_config['base_channels']
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # If using distributed training
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=True
        )
    
    print('#----------Preparing Training----------#')
    # Create optimizer and learning rate scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Get loss function
    criterion = config.criterion
    
    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')
    best_epoch = 0
    step = 0
    
    # If a checkpoint exists, load it
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        logger.info(f'Resume from epoch {start_epoch}')
    
    print('#----------Starting Training----------#')
    # Main training loop
    for epoch in range(start_epoch, config.epochs):
        # Train one epoch
        train_loss, step = train_one_epoch(
            train_loader, model, criterion, optimizer,
            scheduler, epoch, step, logger, config, writer
        )
        
        # Validate
        val_loss = validate_one_epoch(
            val_loader, model, criterion,
            epoch, logger, config, writer
        )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            if config.save_best:
                torch.save(model.state_dict(),
                         os.path.join(config.work_dir, 'checkpoints', 'best.pth'))
        
        # Check before saving checkpoint
        if config.save_last:
            checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
            success = save_checkpoint(
                model, optimizer, scheduler, epoch, step,
                best_loss, best_epoch, checkpoint_dir, logger
            )
            if not success:
                logger.warning("Failed to save checkpoint, continuing training...")
        
        # Clean GPU cache
        if (epoch + 1) % config.clean_cache_interval == 0:
            torch.cuda.empty_cache()
    
    print('#----------Training Complete----------#')
    # Test using the best model
    if os.path.exists(os.path.join(config.work_dir, 'checkpoints', 'best.pth')):
        print('#----------Starting Testing----------#')
        # Load best model
        best_weight = torch.load(
            os.path.join(config.work_dir, 'checkpoints', 'best.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(best_weight)
        
        # Test
        test_loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config
        )
        
        # Rename best model file
        os.rename(
            os.path.join(config.work_dir, 'checkpoints', 'best.pth'),
            os.path.join(config.work_dir, 'checkpoints', 
                        f'best-epoch{best_epoch}-loss{best_loss:.4f}.pth')
        )
        
        # Log final results
        log_info = f'Training complete. Best validation loss: {best_loss:.4f} ' \
                  f'(Epoch {best_epoch}). Test loss: {test_loss:.4f}'
        logger.info(log_info)

def init_distributed_mode(args):
    """ 
    Initialize distributed training settings
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()

def save_checkpoint(model, optimizer, scheduler, epoch, step, best_loss, best_epoch, checkpoint_dir, logger):
    """Safely save checkpoint"""
    try:
        # First check directory permissions
        if not check_write_permissions(checkpoint_dir):
            logger.error(f"No write permissions for directory: {checkpoint_dir}")
            return False
            
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(checkpoint_dir)
        if free < 1024 * 1024 * 100:  # Ensure at least 100MB of free space
            logger.error(f"Not enough disk space. Free space: {free / (1024*1024):.2f} MB")
            return False
            
        # Save to temporary file first
        temp_path = os.path.join(checkpoint_dir, 'temp_latest.pth')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'best_loss': best_loss,
            'best_epoch': best_epoch
        }
        
        # Print checkpoint size
        import sys
        logger.info(f"Checkpoint size: {sys.getsizeof(checkpoint) / (1024*1024):.2f} MB")
        
        # Save to temporary file
        torch.save(checkpoint, temp_path)
        logger.info(f"Successfully saved temporary checkpoint to {temp_path}")
        
        # If temporary file saved successfully, rename to final file
        final_path = os.path.join(checkpoint_dir, 'latest.pth')
        os.replace(temp_path, final_path)
        logger.info(f"Successfully renamed checkpoint to {final_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    # Get configuration
    config = setting_config
    
    # If using distributed training
    if config.distributed:
        init_distributed_mode(config)
    
    # Start training
    main(config)