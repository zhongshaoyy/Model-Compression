class Config:
    log = "./log"  # Path to save log
    checkpoints = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None
    dataset_type = "cifar"
    train_dataset_path = '/public/home/tgz/dataset/CIFAR100/'
    val_dataset_path = '/public/home/tgz/dataset/CIFAR100/'

    num_classes = 100

    epochs = 200
    batch_size = 128
    lr = 0.1
    num_workers = 6

    loss_list = [
        {
            "loss_name": "CELoss",
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "ce_family",
            "loss_rate_decay": "lrdv1"
        },
        # {
        #     "loss_name": "KDLossv2",
        #     "T": 1,
        #     "loss_rate": 0.1,
        #     "factor": 1,
        #     "loss_type": "kdv2_family",
        #     "loss_rate_decay": "lrdv1"
        # },
        {
            "loss_name": "CDLoss",
            "loss_rate": 1,
            "factor": 0.9,
            "loss_type": "fd_family",
            "loss_rate_decay": "lrdv1"
        },
    ]
