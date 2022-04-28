class Chunk:
    # data_field_len -- zmienna otrzymywana za pomoca self.length
    Length_Field_Len = 4
    Type_Field_Len = 4
    CRC_Field_Len = 4

    def __init__(self, length, type_, data, crc):
        self.length = length
        self.type_ = type_
        self.data = data
        self.crc = crc
