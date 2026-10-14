"""Optional, separately pinned visual fixture; scores are replayed from new bars."""
from stock_monitor.io import DataContractError, sha256
from stock_monitor.upgrade.ui_service import load_ui_bundle

SESSION = '2025-02-17'
WAVES_SHA = '644c23671277d12f606135ed411641d19b9a5041308f2e10a8c94f2f5c744231'
LOGOS = {'/logos/FPT.svg': ('FPT.svg', 'image/svg+xml'),
         '/logos/HPG.png': ('HPG.png', 'image/png'),
         '/logos/VCB.png': ('VCB.png', 'image/png')}


class WavesBundle:
    def __init__(self, root):
        self.bundle = load_ui_bundle(root, 'configs/demo_waves.json')
        if self.bundle['key'] != WAVES_SHA or self.bundle['sessions'] != [SESSION]:
            raise DataContractError('Unapproved waves fixture binding')
        paths = [root / 'configs/demo_waves.json', root / 'demo/waves/manifest.json']
        paths.extend(root / 'demo/waves' / row['name'] for row in self.bundle['manifest']['files'])
        self.pins = {p: sha256(p) for p in paths}

    def scope(self, symbol, model):
        if symbol not in {'FPT', 'HPG', 'VCB'} or model != 'v2':
            raise DataContractError('Unknown waves symbol/model')
        if any(not p.is_file() or sha256(p) != digest for p,digest in self.pins.items()):
            raise DataContractError('Waves fixture checksum mismatch')
        return self.bundle
