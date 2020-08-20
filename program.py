import os
import test_cv

# Удалим предыдущие результаты
path = 'C:\\git\\bi\\sample'

for f in os.listdir(path):
    if f.find('detected') > -1:
        os.remove(os.path.join(path, f))

images, labels = test_cv.detect_face(path)

