import pytorch_lightning as pl

import ray

from tablestakes.ml import model_transformer, hyperparams, torch_helpers

num_gpus = 1

ray.init(
    address='auto',
    ignore_reinit_error=True,
)


@ray.remote(num_gpus=num_gpus, max_calls=1)
def run_one(hp: hyperparams.LearningParams):
    net = model_transformer.RectTransformerModule(hp)

    pl_callbacks = [
        torch_helpers.CounterTimerCallback(),
    ]

    trainer = pl.Trainer(
        logger=torch_helpers.get_pl_logger(hp),
        callbacks=pl_callbacks,
        max_epochs=hp.num_epochs,
        weights_summary='full',
        profiler=True,
        gpus=num_gpus,
    )

    print("Starting trainer.fit:")
    trainer.fit(net)

    print('done! with fit')

    return True


if __name__ == '__main__':
    dataset_name = 'num=10000_99e0'

    encoder_types = [
        'torch',
        'fast_default',
        'fast_favor',
        'fast_grf',
        'performer',
        'ablatable_do_drop_k',
        'ablatable_do_not_drop_k',
    ]

    hp = hyperparams.LearningParams(dataset_name)
    hp.num_epochs = 10
    hp.lr = 0.001

    for encoder_type in encoder_types:
        print('')
        print(f'Starting {encoder_type}')
        hp.trans_encoder_type = encoder_type
        hp.experiment_tags = ['test', 'benchmark_encoder']
        run_one.remote(hp)

    print('done')
