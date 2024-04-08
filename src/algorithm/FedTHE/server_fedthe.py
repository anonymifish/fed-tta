from src.algorithm.Base.server_base import BaseServer


class FedTHEServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super(FedTHEServer, self).__init__(device, backbone, configs)
