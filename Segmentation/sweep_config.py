
sweep_config = {
    'method': 'bayes',  # Bayesien
    'metric': {
        'name': 'val_loss',  # métrique à optimiser
        'goal': 'minimize'
    },
    'parameters': {
        # Batch size à tester
        'training.batch_size': {
            'values': [2, 4, 8]
        },
        # Optimizer
        'training.optimizer': {
            'values': ['SGD', 'Adam']
        },
        # Learning rate
        'training.lr': {
            'min': 1e-4,
            'max': 1e-2
        },
        # Scheduler type
        'training.scheduler': {
            'values': ['linear', 'cosine']
        },
        # Probabilité d'augmentation
        'augmentation.prob': {
            'min': 0.0,
            'max': 0.5
        }
    }
}