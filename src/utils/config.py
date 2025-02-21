def get_config():
    config = {
        'batch_size': 32,
        'lr': 0.0068,
        'epochs': 20,
        'data_dir': "/kaggle/input/11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/cls_data/",
        'data_ver_dir': "/kaggle/input/11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/ver_data/",
        'checkpoint_dir': "working/",
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'num_workers': 8,
        'pin_memory': True,
        'scheduler_step_size': 5,
        'scheduler_gamma': 0.1,
        'num_classes': 8631,
        'weight_decay': 1e-2
    }
    return config