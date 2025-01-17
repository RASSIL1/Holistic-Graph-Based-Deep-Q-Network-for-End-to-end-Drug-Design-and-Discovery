config = {
    'data_path': 'data/processed/zinc_plogp_sorted.csv',
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'eval_frequency': 10,
    'epsilon': 0.1,
    'gamma': 0.99,
    'replay_buffer_size': 10000,
    'target_update_frequency': 10,
    'similarity_threshold': 0.4,
    'ggin': {
        'in_feats': 34,
        'h_feats': 128,
        'num_layers': 3,
        'num_classes': 4
    },
    'dqn': {
        'in_feats': 128,
        'h_feats': 128,
        'num_actions': 4
    }
}
