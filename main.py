from binascii import crc32


class PNG:

    def __init__(self, fname=None):
        self.width = 0
        self.height = 0
        self.bit_depth = 0
        self.color_type = 0
        self.chunks = []
        self.pallet = []
        self.funcs = {}
        self.funcs[b'IHDR'] = self._ihdr_chunk
        self.funcs[b'sRGB'] = self._srgb_chunk
        self.funcs[b'gAMA'] = self._gama_chunk
        self.funcs[b'tRNS'] = self._trns_chunk
        self.funcs[b'PLTE'] = self._plte_chunk
        self.funcs[b'pHYs'] = self._phys_chunk
        self.funcs[b'tEXt'] = self._text_chunk
        if fname:
            self.read_png(fname)

    def _read_chunks(self, fname):
        self.chunks.clear()
        with open(fname, 'rb') as a_file:
            if a_file.read(8) != b'\x89PNG\r\n\x1a\n':
                raise Exception('Signature error')
            datl = a_file.read(4)
            while datl != b'':
                length = int.from_bytes(datl, 'big')
                data = a_file.read(4 + length)
                if int.from_bytes(a_file.read(4), 'big') != crc32(data):
                    raise Exception('CRC Checkerror')
                self.chunks.append([data[:4], data[4:]])
                datl = a_file.read(4)

    def _def_chunk(self, index, name, data):
        print(f'+ {name.decode("utf-8")} Chunk')
        _ = index
        _ = data

    def _srgb_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        _ = {
            0: 'Perceptual',
            1: 'Relative colorimetric',
            2: 'Saturation',
            3: 'Absolute colorimetric'
        }
        index = int.from_bytes(data, 'big')
        print(f'  + Rendering intent: {_[index]}')

    def _gama_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        print(f'  + Image gamma: {int.from_bytes(data, "big")}')

    def _plte_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index, data
        _ = divmod(len(data), 3)
        if _[1] != 0:
            raise Exception('pallet')
        for pin in range(_[0]):
            color = data[pin * 3], data[pin * 3 + 1], data[pin * 3 + 2]
            print(f'  - #{color[0]:02X}{color[1]:02X}{color[2]:02X}')
            self.pallet.append(color)

    def _trns_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        length = len(data)
        if self.color_type == 0:
            if length != 2:
                raise Exception('format error')
            print(f'  - Grey sample value: {int.from_bytes(data, "big")}')
        elif self.color_type == 2:
            if length != 6:
                raise Exception('format error')
            print(f'  - R sample value: {int.from_bytes(data[0:2], "big")}')
            print(f'  - B sample value: {int.from_bytes(data[2:4], "big")}')
            print(f'  - G sample value: {int.from_bytes(data[4:6], "big")}')
        elif self.color_type == 3:
            for a_i in range(len(self.pallet)):
                _ = self.pallet[a_i]
                print(f'  - {data[a_i]:02X}(#{_[0]:02X}{_[1]:02X}{_[2]:02X})')

    def _phys_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        _ = data[:4], data[4:8], data[8:9]
        spec = ''
        if _[2] == b'\01':
            spec = 'px/m'
        print(f'  - Pixel per unit, X axis {int.from_bytes(_[0], "big")}' + spec)
        print(f'  - Pixel per unit, Y axis {int.from_bytes(_[1], "big")}' + spec)

    def _text_chunk(self, index, name, data):
        _ = index, name
        print(f'- {name.decode("utf-8")} Chunk')
        _ = data.split(b'\x00')
        print(f'  + {_[0].decode("utf-8")}: {_[1].decode("utf-8")}')

    def _ihdr_chunk(self, index, name, data):
        if index != 0:
            raise Exception('first chunk')
        _ = name
        self.width = int.from_bytes(data[:4], 'big')
        self.height = int.from_bytes(data[4:8], 'big')
        self.bit_depth = int.from_bytes(data[8:9], 'big')
        self.color_type = int.from_bytes(data[9:10], 'big')
        print('- IHDR Chunk')
        print(f'  + width : {self.width}')
        print(f'  + height: {self.height}')
        print(f'  + bit depth: {self.bit_depth}')
        _ = {
            0: 'Grayscale',
            2: 'Truecolor',
            3: 'Indexed-color',
            4: 'Greyscale with alpha',
            6: 'Truecolor with alpha'}
        print(f'  + color type: {_[self.color_type]}')
        print(f'  + compression method: {int.from_bytes(data[10:11], "big")}')
        print(f'  + filter method: {int.from_bytes(data[11:12], "big")}')
        print(f'  + interlace method: {int.from_bytes(data[12:13], "big")}')

    def read_png(self, fname):
        """Loading png images"""
        self._read_chunks(fname)
        for index, chunk in enumerate(self.chunks):
            self.funcs.get(chunk[0], self._def_chunk)(index, chunk[0], chunk[1])


if __name__ == '__main__':
    PPP = PNG('test2.png')
