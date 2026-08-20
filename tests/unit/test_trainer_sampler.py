import json
from types import SimpleNamespace

from siete_rl.trainer import SWEGRPOTrainer


def _training_sampler(checkpoint=None):
    trainer = object.__new__(SWEGRPOTrainer)
    trainer._preloaded_checkpoint = checkpoint
    trainer.train_dataset = list(range(12))
    trainer.num_generations = 1
    trainer.num_iterations = 1
    trainer.shuffle_dataset = True
    trainer.args = SimpleNamespace(
        generation_batch_size=2,
        steps_per_generation=1,
        seed=42,
    )
    return trainer._get_train_sampler()


def test_training_sampler_recreates_epoch_order_for_resume(tmp_path) -> None:
    first_epoch = _training_sampler()
    original = _training_sampler()
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"epoch": 2.5}), encoding="utf-8"
    )
    resumed = _training_sampler(tmp_path)

    assert hasattr(original, "set_epoch")
    first_epoch.set_epoch(0)
    original.set_epoch(2)
    resumed.set_epoch(0)

    first_order = list(first_epoch)
    original_order = list(original)
    resumed_order = list(resumed)
    assert original_order == resumed_order
    assert first_order != original_order
