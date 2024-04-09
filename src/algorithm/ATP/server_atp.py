from src.algorithm.Base.server_base import BaseServer


class ATPServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)

    def fit(self):
        for r in range(self.global_rounds):
            pass

    def test(self):
        pass
