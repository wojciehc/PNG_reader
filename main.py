import random
from binascii import crc32

import sympy
import cv2
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv, rgba2rgb


# nwd
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


#odwrotnosc
def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi

    while e > 0:
        temp1 = temp_phi / e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2

        x = x2 - temp1 * x1
        y = d - temp1 * y1

        x2 = x1
        x1 = x
        d = y1
        y1 = y

    if temp_phi == 1:
        return d + phi


def keyGenerator(p, q):
    if not(sympy.isprime(q) and sympy.isprime(p)):
        raise ValueError('ktoras z liczb nie jest pierwsza')
    elif p == q:
        raise ValueError('p i q nie moga byc takie same')

    n = p*q
    phi = (p-1) * (q-1)

    e = random.randrange(1, phi)

    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)

    d = multiplicative_inverse(e, phi)
    return (e, n), (d, n)


def encrypt(public_key, text):
    key, n = public_key
    cipher = [(ord(char) ** key) % n for char in text]

    return cipher


def decrypt(public_key, text):
    key, n = public_key
    decrypted = [chr((char ** key) % n) for char in text]

    return ''.join(text)


class PNG:
    SIGNATURE = b'\x89PNG\r\n\x1a\n'

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
        with open(fname, 'rb') as _file:
            if _file.read(8) != PNG.SIGNATURE:
                raise Exception('Signature error')
            _data = _file.read(4)
            while _data != b'':
                length = int.from_bytes(_data, 'big')
                data = _file.read(4 + length)
                if int.from_bytes(_file.read(4), 'big') != crc32(data):
                    raise Exception('CRC error')
                self.chunks.append([data[:4], data[4:]])
                _data = _file.read(4)

    def _def_chunk(self, index, name, data):
        print(f'+ {name.decode("utf-8")} Chunk')
        _ = index
        _ = data

    def _srgb_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        _ = {
            0: 'Perceptualna',
            1: 'Kolorymetria względna',
            2: 'Saturacja',
            3: 'Kolorymetria absolutna'
        }
        index = int.from_bytes(data, 'big')
        print(f'  + Metoda renderowania:: {_[index]}')

    def _gama_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        print(f'  + Gamma obrazu: {int.from_bytes(data, "big")}')

    def _plte_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index, data
        _ = divmod(len(data), 3)
        if _[1] != 0:
            raise Exception('pallet error')
        for pin in range(_[0]):
            color = data[pin * 3], data[pin * 3 + 1], data[pin * 3 + 2]
            print(f'  - #{color[0]:02X}{color[1]:02X}{color[2]:02X}')
            self.pallet.append(color)

    def _trns_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        length = len(data)
        if self.color_type == 0: #szarosc
            if length != 2:
                raise Exception('format error')
            print(f'  - Wartość szarej próbki: {int.from_bytes(data, "big")}')
        elif self.color_type == 2: #true color
            if length != 6:
                raise Exception('format error')
            print(f'  - Wartość czerwonej (R) próbki: {int.from_bytes(data[0:2], "big")}')
            print(f'  - Wartość niebieskiej (B) próbki: {int.from_bytes(data[2:4], "big")}')
            print(f'  - Wartość zielonej (G) próbki: {int.from_bytes(data[4:6], "big")}')
        elif self.color_type == 3: #kolor indeksowany
            for a_i in range(len(self.pallet)):
                _ = self.pallet[a_i]
                #print(f'  - {data[a_i]:02X}(#{_[0]:02X}{_[1]:02X}{_[2]:02X})')

    def _phys_chunk(self, index, name, data):
        print(f'- {name.decode("utf-8")} Chunk')
        _ = index
        _ = data[:4], data[4:8], data[8:9]
        unit = ''
        if _[2] == b'\01':
            spec = 'px/m'
        print(f'  - Pixel na jednostke, os X {int.from_bytes(_[0], "big")}' + unit)
        print(f'  - Pixel na jednostke, os Y {int.from_bytes(_[1], "big")}' + unit)

    def _text_chunk(self, index, name, data):
        _ = index, name
        print(f'- {name.decode("utf-8")} Chunk')
        _ = data.split(b'\x00')
        print(f'  + {_[0].decode("utf-8")}: {_[1].decode("utf-8")}')

    def _ihdr_chunk(self, index, name, data):
        if index != 0:
            raise Exception('first chunk')

        self.width = int.from_bytes(data[:4], 'big')
        self.height = int.from_bytes(data[4:8], 'big')
        self.bit_depth = int.from_bytes(data[8:9], 'big')
        self.color_type = int.from_bytes(data[9:10], 'big')
        print('- IHDR Chunk')
        print(f'  + szerokosc : {self.width}')
        print(f'  + wysokosc: {self.height}')
        print(f'  + glebia bitowa: {self.bit_depth}')
        _ = {
            0: 'Skala szarosci',
            2: 'Truecolor',
            3: 'Kolor indeksowany',
            4: 'Skala szarosci z kanalem alfa',
            6: 'Truecolor z kanalem alfa'}
        print(f'  + typ przestrzeni bitowej: {_[self.color_type]}')
        print(f'  + metoda kompresji: {int.from_bytes(data[10:11], "big")}')
        print(f'  + metoda filtrowania: {int.from_bytes(data[11:12], "big")}')
        print(f'  + metoda przeplotu: {int.from_bytes(data[12:13], "big")}')

    def read_png(self, fname):
        self._read_chunks(fname)
        for index, chunk in enumerate(self.chunks):
            self.funcs.get(chunk[0], self._def_chunk)(index, chunk[0], chunk[1])

    def save_png(self, fname, chunks_to_remove = None):
        with open(fname, 'wb') as file:
            file.write(PNG.SIGNATURE)
            for chunk in self.chunks:
                if chunks_to_remove is None or chunk[0] not in chunks_to_remove:
                    file.write((len(chunk[1])).to_bytes(4, 'big'))
                    file.write(chunk[0])
                    file.write(chunk[1])
                    file.write(crc32(chunk[0] + chunk[1]).to_bytes(4, 'big'))

    def crypt_png(self, fname, chunks_to_encrypt = None):
        public, private = keyGenerator(sympy.randprime(0, 1000000), sympy.randprime(0, 1000000))
        with open(fname, 'wb') as file:
            file.write(PNG.SIGNATURE)
            for chunk in self.chunks:
                data = chunk[1]
                if chunk[0] in chunks_to_encrypt:
                    data = encrypt(public, chunk[1])

                file.write((len(data)).to_bytes(4, 'big'))
                file.write(chunk[0])
                file.write(data)
                file.write(crc32(chunk[0] + data).to_bytes(4, 'big'))


if __name__ == '__main__':
    print('Chunki w pliku przed animizacja:')

    # TUTAJ NALEZY WPISAC NAZWE PLIKU
    image_name = 'test.png'


    png_file = PNG(image_name)



    #Tablica chunkow ktorych chcemy sie pozbyc podczas procesu animizacji

    # chunks = [b'eXIf', b'tEXt', b'tIME', b'zTXt', b'iTXt', b'dSIG', b'gAMA', b'pHYs', b'iCCP', b'bKGD', b'sBIT',
    #           b'tRNS', b'cHRM', b'sRGB', b'iCCP', b'KGD', b'sPLT', b'hIST']
    # png_file.save_png('out.png', chunks)
    #
    # print('Chunki w pliku po animizacji:')
    # cleared_png = PNG('out.png')
    #
    # image = rgba2rgb(cv2.imread(image_name, cv2.IMREAD_UNCHANGED))
    # dark_image = rgb2gray(image)
    # imsave("grey.png", dark_image)
    #
    # dark_image_fft = np.fft.fftshift(np.fft.fft2(dark_image))
    # inverse_fft = np.fft.ifft2(np.fft.ifftshift(dark_image_fft))
    # ift_real = inverse_fft.real
    # imsave("angle.png", np.angle(dark_image_fft))
    # imsave("fft.png", np.log(abs(dark_image_fft)))
    # imsave("ifft.png", ift_real)
    # from PIL import Image
    # im = Image.open('out.png')
    # im.show()
    #
    #
    # Image.open("grey.png").show()
    # Image.open('fft.png').show()
    # Image.open('ifft.png').show()
    # Image.open('angle.png').show()

