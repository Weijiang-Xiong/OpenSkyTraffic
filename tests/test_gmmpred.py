import torch
import pickle

from netsanut.models import GMMPred, HiMSNet

with open("tests/simbarca_batch.pkl", "rb") as f:
    batch = pickle.load(f)

model = GMMPred(adjacency_hop=5, zero_init=True)
model.train()
model.adapt_to_metadata(batch["metadata"])
loss_dict = model(batch)
print(loss_dict.keys())
loss = sum(loss_dict.values())
loss.backward()
print("Loss:", loss.item())

model.eval()
pred = model(batch)
print(pred.keys())

state_dict = model.state_dict() 

# now activate aleatoric uncertainty
model_prob = GMMPred(adjacency_hop=5)
model_prob.train()
model_prob.adapt_to_metadata(batch["metadata"]) # this should do nothing, as the metadata is already set 

loss_dict = model_prob(batch)
print(loss_dict.keys())
loss = sum(loss_dict.values())
loss.backward()
print("Loss:", loss.item())

model_prob.eval()
pred = model_prob(batch)
print(pred.keys())