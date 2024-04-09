from src.algorithm.Base.client_base import BaseClient


class ATPClient(BaseClient):
    def __init__(self, cid, device, backbone, confgis):
        super(ATPClient, self).__init__(cid, device, backbone, confgis)

    def train(self):
        pass

    def test(self):
        pass
