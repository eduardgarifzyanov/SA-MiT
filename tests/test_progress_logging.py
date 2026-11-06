import logging

import torch
from torch.utils.data import DataLoader

from samit.training.trainer import TwoStageTrainer
from samit.training.optim import WarmupCosineScheduler


def test_progress_reports_final_partial_interval(caplog):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(1, 2, 1)

        def forward(self, x, mode='seg'):
            return self.layer(x)

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=0,
                                    total_steps=3, base_lr=0.01, min_lr=0.001)
    loader = DataLoader([{'image': torch.ones(1, 2, 2),
                          'label': torch.zeros(2, 2, dtype=torch.long)}] * 3,
                        batch_size=1)
    trainer = TwoStageTrainer(model, optimizer, scheduler,
                              torch.nn.CrossEntropyLoss(), loader,
                              device='cpu', use_amp=False, log_interval=2,
                              progress_logger=logging.getLogger('test.progress'))
    with caplog.at_level(logging.INFO, logger='test.progress'):
        result = trainer.train_one_epoch(0)
    assert result['loss'] > 0
    assert len(caplog.records) == 2
    assert 'batch=2/3' in caplog.records[0].message
    assert 'batch=3/3 progress=100.00%' in caplog.records[1].message
    assert 'epoch_eta_seconds=0' in caplog.records[1].message
