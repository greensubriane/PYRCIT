import wradlib as wrl
import numpy as np

from matplotlib import pyplot as plt

# fpath = 'G:/RZ/raa01-ry_10000-2112181350-dwd---bin.gz'
# f = wrl.util.get_wradlib_data_file(fpath)
data, metadata = wrl.io.read_radolan_composite("G:/RZ/raa01-ry_10000-2112181350-dwd---bin")

print(data.shape)
print(metadata.keys())

maskeddata = np.ma.masked_equal(data, metadata["nodataflag"])

fig = plt.figure(figsize=(10, 8))
# get coordinates
radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
x = radolan_grid_xy[:, :, 0]
y = radolan_grid_xy[:, :, 1]

# create quick plot with colorbar and title
plt.figure(figsize=(10, 8))
plt.pcolormesh(x, y, maskeddata)
plt.show()