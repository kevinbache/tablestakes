import pytorch_lightning as pl

import ray

from tablestakes.ml import model_transformer, hyperparams, metrics_mod

num_gpus = 1
num_cpus = 2

ray.init(
    address='auto',
    ignore_reinit_error=True,
)


@ray.remote(num_gpus=num_gpus, num_cpus=num_cpus, max_calls=1)
def run_one(hp: hyperparams.LearningParams):
    net = model_transformer.RectTransformerModule(hp)

    pl_callbacks = [
        metrics_mod.CounterTimerCallback(),
    ]

    trainer = pl.Trainer(
        logger=metrics_mod.get_pl_logger(hp),
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
        # 'fast_default',
        # 'fast_favor',
        # 'fast_grf',
        # # 'performer',
        # 'ablatable_do_drop_k',
        # 'ablatable_do_not_drop_k',
    ]
    encoder_types.reverse()

    hp = hyperparams.LearningParams(dataset_name)
    hp.num_epochs = 10
    hp.lr = 0.001

    print('')
    outs = []
    for encoder_type in encoder_types:
        print(f'Starting {encoder_type}')
        hp.trans_encoder_type = encoder_type
        hp.experiment_tags = ['encoder_benchmark_v2-reverse']
        outs.append(run_one.remote(hp))

    print(ray.get(outs))
    print('done')
