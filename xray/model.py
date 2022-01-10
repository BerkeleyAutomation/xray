from torchvision.models.segmentation import fcn_resnet50


class XrayModel:
    """
    A thin wrapper around a fully convolutional resnet model
    """

    def __init__(self, ratios):
        self._model = fcn_resnet50(num_classes=len(ratios))

    def __call__(self, input):
        return self._model(input)

    def load(self, state_dict):
        self._model.load_state_dict(state_dict)

    @property
    def training(self):
        return self._model.training

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def to(self, device):
        self._model = self._model.to(device)
        return self

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()
