class MatrixHash:
    def __call__(self, matrix):
        seed = 0
        data = matrix.flatten()
        for elem in data:
            # Replicate C++'s hash combining formula
            h = hash(elem)
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2)
        return seed
