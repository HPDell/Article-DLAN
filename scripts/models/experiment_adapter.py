"""Construct the saved article configurations with unchanged model operations."""
from torch import nn
from .DLAN import DeepGWRTrainer, DLAN, TrainerConfig
from .adapters import MatchedCaseTrainer, MatchedGeoAggregator, MatchedGCNNWR

class ArticleTrainer(MatchedCaseTrainer):
    """Apply the shared train-only scaling, retrieval and fitting implementation."""
    def __init__(self, specification: dict, config: TrainerConfig):
        architecture = specification['architecture']
        kind = specification['model']
        if kind == 'dlan':
            width, first, second = architecture['dlan']
            model = DLAN(architecture['x_dim'], architecture['coordinate_dim'],
                         d_model=width, num_heads=1, num_blocks=1, dropout=0.1,
                         local_hidden_dims=(first, second))
        elif kind == 'geoaggregator':
            model = MatchedGeoAggregator(architecture['x_dim'], architecture['coordinate_dim'],
                                         tuple(architecture['geo']))
        elif kind == 'gcnnwr':
            model = MatchedGCNNWR(architecture['x_dim'], tuple(architecture['gcnn']))
        else:
            raise ValueError(kind)
        DeepGWRTrainer.__init__(self, model, config)
        self.policy = kind
        self.candidate_count = specification['candidates']
        self.microbatch_size = specification['microbatch_size']
        self.progress_path = None
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != specification['parameters']:
            raise ValueError((count, specification['parameters']))

def build_trainer(specification: dict, device: str = 'cpu'):
    """Create a regressor/trainer; import the standard GNNWR library only if needed."""
    config = TrainerConfig(**{**specification['training'], 'device': device})
    if specification['model'] in ('gnnwr', 'gtnnwr'):
        from .GNNWR import UpstreamGNNWRTrainer
        return UpstreamGNNWRTrainer(specification['case_key'], config,
                                   microbatch_size=specification['microbatch_size'])
    return ArticleTrainer(specification, config)
