sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'training_batch_size': {'values': [2, 4, 8]},
        'training_learning_rate': {'min': 1e-4, 'max': 1e-2},
        'augmentation_spatial_size': {
            'values': ['(96, 96, 64)', '(96, 96, 96)']
        }
    }
}