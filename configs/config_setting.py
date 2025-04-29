from torchvision import transforms
from utils import *
from datetime import datetime
import os

class setting_config:
    """
    Configuration class for training settings
    """
    # Model configuration
    network = 'mscbunet'  # Network model used
    model_config = {
        'base_channels': 32,        # Base number of channels
        'reduction_ratio': 8,       # Channel compression ratio
        'min_spatial_size': 8,      # Minimum feature map size
        'downsample_ratio': 4,      # Spatial downsampling ratio
        'num_heads': 4,
        'kernel_size': 3,
        'attn_drop': 0.1,
        'proj_drop': 0.1,
    }

    # Dataset configuration
    project_root = '/autodl-fs/data/MSCB-UNet/'
    # Dataset configuration
    datasets = 'isic18'
    if datasets == 'isic17':
        data_path = os.path.join(project_root, 'data/isic2017')
    elif datasets == 'isic18':
        data_path = os.path.join(project_root, 'data/isic2018/')
    else:
        raise Exception('datasets in not right!')
    
    input_channels = 3             # Number of input channels
    num_classes = 1                # Number of classes
    input_size_h = 256            # Input height
    input_size_w = 256            # Input width

    # Training configuration
    epochs = 200                  # Total number of epochs
    batch_size = 4                 # Keep small batch size
    num_workers = 2                 # Temporarily set to single process
    print_interval = 10           # Print interval
    save_interval = 50   

    # Modify working directory configuration
    if os.path.exists('/root/autodl-tmp'):  # AutoDL environment
        # Create a symbolic link to the original directory
        original_base_dir = '/autodl-fs/data/MSCB-UNet/results'
        new_base_dir = '/root/autodl-tmp/MSCB-UNet/results'
        
        # Ensure the new directory exists
        os.makedirs(new_base_dir, exist_ok=True)
        
        # Set the base directory to the new location
        base_dir = new_base_dir
        
        # Create a symbolic link (if it does not exist)
        if not os.path.exists(original_base_dir):
            os.makedirs(os.path.dirname(original_base_dir), exist_ok=True)
            try:
                os.symlink(new_base_dir, original_base_dir)
            except Exception as e:
                print(f"Warning: Could not create symlink: {e}")
    else:
        # Use original path in non-AutoDL environment
        base_dir = '/autodl-fs/data/MSCB-UNet/results'

    # Working directory configuration
    work_dir = os.path.join(base_dir, 
                           f'{network}_{datasets}_{datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")}/')
    gpu_id = '0'                  # GPU ID
    seed = 42                     # Random seed
    distributed = False           # Whether to use distributed training

    # Data augmentation configuration
    train_transformer = transforms.Compose([
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[-30, 30]),
        myNormalize(datasets, train=True),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])
    
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    # Optimizer configuration
    opt = 'AdamW'                # Use AdamW optimizer
    lr = 1e-4                    # Learning rate
    betas = (0.9, 0.999)         # Beta parameters for Adam/AdamW
    eps = 1e-8                   # Numerical stability parameter
    weight_decay = 1e-2          # Weight decay
    amsgrad = False              # Whether to use AMSGrad variant

    # Learning rate scheduler configuration
    sch = 'CosineAnnealingLR'    # Use cosine annealing
    T_max = 50                   # Adjustment period
    eta_min = 1e-6               # Minimum learning rate
    last_epoch = -1              # Last epoch
    warm_up_epochs = 10          # Warm-up epochs

    # Loss function configuration
    criterion = GT_BceDiceLoss(wb=1.0, wd=1.0)  # Use combined loss function
    aux_loss_weights = {         # Auxiliary loss weights
        'level5': 0.5,
        'level4': 0.4,
        'level3': 0.3,
        'level2': 0.2,
        'level1': 0.1
    }

    # Training optimization configuration
    gradient_accumulation_steps = 8  # Increase gradient accumulation steps
    max_grad_norm = 1.0             # Gradient clipping threshold
    amp = True                      # Enable automatic mixed precision

    # Memory optimization configuration
    pin_memory = True              # Keep enabled
    prefetch_factor = 2            # Keep small prefetch factor
    persistent_workers = True      # Disable persistent workers
    clean_cache_interval = 5        # Cache cleaning interval
    tensorboard_log_freq = 20       # Tensorboard logging frequency
    empty_cache_freq = 50           # Frequency to clean GPU cache

    # Model checkpoint configuration
    use_checkpoint = True           # Use gradient checkpointing
    save_best = True               # Whether to save the best model
    save_last = True               # Whether to save the latest model

    # Validation and testing configuration
    val_interval = 1               # Validation interval
    test_interval = 5              # Testing interval

    # Add new memory optimization configuration
    multiprocessing_context = 'spawn'  # Use spawn method to create processes
    worker_init_fn = None             # Do not use worker initialization function

    
