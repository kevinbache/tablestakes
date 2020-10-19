import pytorch_lightning as pl

import ray

from tablestakes.ml import model_transformer, hyperparams

num_gpus = 1

ray.init(
    address='auto',
    ignore_reinit_error=True,
)


@ray.remote(num_gpus=num_gpus)
def run_one():
    dataset_name = 'num=10000_99e0'

    hp = hyperparams.LearningParams(dataset_name)
    hp.do_include_embeddings = True
    net = model_transformer.RectTransformerModule(hp)

    trainer = pl.Trainer(
        # logger=torch_helpers.get_pl_logger(hp),
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
    out = run_one.remote()
    print(ray.get(out))
    print('done')
