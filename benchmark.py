from models import simpleCNN, shufflenet, mixnet
from torchsummary import summary
import torch
from torch.autograd import Variable
import timeit


def benchmark_model(model, iters, dim):
    dummy_data = iters * [Variable(torch.rand(dim))]
    model.eval()
    start = timeit.default_timer()
    with torch.no_grad():
        for image in dummy_data:
            output = model(image)
    end = timeit.default_timer()
    runtimeCPU = (end-start)/iters

    model.cuda()
    model.eval()
    dummy_data = iters * [Variable(torch.rand(dim).cuda())]
    start = timeit.default_timer()
    with torch.no_grad():
        for image in dummy_data:
            output = model(image)
    end = timeit.default_timer()
    runtimeGPU = (end-start)/iters

    model.train()
    summary(model,dim[1:])
    print("Runtime GPU: ", runtimeGPU)
    print("Runtime CPU: ", runtimeCPU)




test_iters = 50
dim=(1,1,28,28)

print ("Benchmarking Simple CNN")
model=simpleCNN()
benchmark_model(model,test_iters,dim)
print ("Benchmarking Shufflenet v2")
model=shufflenet()
benchmark_model(model,test_iters,dim)
print ("Benchmarking MixNet")
model=mixnet()
benchmark_model(model,test_iters,dim)
