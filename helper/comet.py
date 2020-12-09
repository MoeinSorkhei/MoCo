from comet_ml import Experiment, ExistingExperiment


class CometTracker:
    def __init__(self, comet_params, experiment_name=None, run_params=None):
        self.experiment = Experiment(**comet_params)

        if run_params is not None:
            self.experiment.log_parameters(run_params)

        if experiment_name is not None:
            self.experiment.set_name(experiment_name)

    def track_metric(self, metric, value, step=None):
        self.experiment.log_metric(metric, value, step)

    def add_tags(self, tags):
        self.experiment.add_tags(tags)
        print(f'In [add_tags]: Added these tags to the new experiment: {tags}')

    def set_name(self, name):
        self.experiment.set_name(name)


def init_comet(experiment_name=None):
    # params for Moein
    comet_params = {
        'api_key': "QLZmIFugp5kqZjA4XE2yNS0iZ",
        'project_name': "moco",
        'workspace': "moeinsorkhei"
    }
    tracker = CometTracker(comet_params, experiment_name=experiment_name)
    return tracker
