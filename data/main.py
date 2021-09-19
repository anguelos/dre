from dataset import MultilineStringDataset
from dataloader import MultilineStringDataLoader


# load the dataset
ds = MultilineStringDataset("multilineString.txt")

# load the data loader
dl = MultilineStringDataLoader(dataset=ds, batch_size=4, shuffle=True)


# for i in range(len(ds)):
#     print(ds[i])
# dlr = DataLoader(ds.tensors)
# print(len(ds.output))