import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler


# class Infinitecycle:
#     '''Used to cycle over an iterable'''

#     def __init__(self, create_iterable):
#         self.create_iterable = create_iterable
#         self.iterable = iter(create_iterable)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         try:
#             return next(self.iterable)
#         except StopIteration:
#             self.iterable = iter(self.create_iterable)
#             return next(self)


def logexp(x,th):#ensure  logexp(x)=log(th+exp(x))
    a = np.log(th)
    return F.softplus(x-a)+a

def get_rng_state():
    ''' returns the state of the RNGs'''
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cudarng_state = torch.cuda.get_rng_state()
        return (rng_state, cudarng_state)
    return (rng_state,)


def set_rng_state(state):
    ''' modifiy the state of the RNGs'''
    torch.set_rng_state(state[0])
    if len(state) == 2:
        torch.cuda.set_rng_state(state[1])


class EarlyStop:
    ''''Used to stop the training if the loss does not decrease during [patience] iterations'''

    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.nbest = 0
        self.niter = -1

    def step(self, y):
        self.niter += 1
        if self.best is None:
            self.nbest = 0
            self.best = y
        elif self.best < y:
            self.nbest += 1
        else:
            self.best = y
            self.nbest = 0
        return self.nbest >= self.patience

class DataPoint(torch.utils.data.dataset.Dataset):
    def __init__(self,df,conv):
        super().__init__()
        self.length = df.shape[0]
        self.tensors = conv(df)
#        self.tensors = tuple(x.pin_memory() for x in self.tensors)
    def __getitem__(self,index):
        raise Exception("not working")
    def get_items(self, indices, pin_memory=False):
        batch_index = torch.LongTensor(indices)
        res = tuple(tensor.index_select(0, batch_index) for tensor in self.tensors)
        if pin_memory and torch.cuda.is_available():
            res = tuple(x.pin_memory() for x in res)
        return res
    def __len__(self):
        return self.length


def groupindex(vec):
    l=[]
    vj=None
    for i,vi in enumerate(vec):
        if vj is None or vi!=vj:
            l.append([i])
        else:
            l[-1].append(i)
        vj=vi
    return l

class DataTraj(torch.utils.data.dataset.Dataset):
    def __init__(self,df,conv,idtraj="segment"):
        super().__init__()
        self.grouped=list(torch.LongTensor(x) for x in groupindex(df.loc[:,idtraj].values))
        self.length = len(self.grouped)
        self.longest_traj = max(x.shape[0] for x in self.grouped)
        num_lines = sum(x.shape[0] for x in self.grouped)
        mean_length = num_lines/self.length
        print("num_lines",num_lines,"mean_length",mean_length,"self.longest_traj",self.longest_traj,"#traj", len(self.grouped))
        self.tensors = conv(df)
        if torch.cuda.is_available():
            self.tensors = tuple(tensor.pin_memory() for tensor in self.tensors)
    def __getitem__(self,index):
        raise NotImplementedError
#        return self.ltraj[index]

    def get_items(self, indices, pin_memory=False):
#        print(indices)
        trajs = tuple(self.grouped[i] for i in indices)
#        batch_index = torch.cat(trajs)
        def apply(x):
                return [x.index_select(0, index) for index in trajs]
            # else:
            #     return x.index_select(0, batch_index)
        res = tuple(apply(tensor) for tensor in self.tensors)
        # if pin_memory and torch.cuda.is_available():
        #     res = tuple([xi.pin_memory() for xi in x] for x in res)
        return res
    def __len__(self):
        return self.length


class Infinitecycle:
    '''Used to cycle over an iterable'''
    def __init__(self, create_iterable, nb_ele):
        self.create_iterable = create_iterable
        self.iterable = iter(create_iterable)
        self.count = 0
        self.nb_ele = nb_ele

    def __iter__(self):
 #       print('exec iter',self.count)
#        self.count=0
        return self

    def __next__(self):
#        print('exec next', self.count)
        if self.count == self.nb_ele:
            self.count = 0
            raise StopIteration
        else:
            try:
                n = next(self.iterable)
                self.count += 1
                return n
            except StopIteration:
                self.iterable = iter(self.create_iterable)
                return next(self)
    def __len__(self):
        return self.nb_ele#-self.count


class Select_batchnew:
    def __init__(self, dataset, pin_memory, indexesiterator):
        self.liter = list(indexesiterator)
        self.liter.reverse()
        self.dataset =dataset
        self.pin_memory =pin_memory
    def __iter__(self):
        return self
    def __next__(self):
        if self.liter == []:
            raise StopIteration
        else:
            return self.dataset.get_items(self.liter.pop(), self.pin_memory)
def select_batchnew(dataset, pin_memory, indexesiterator):
    ''' utility function for TensorDataLoader'''
    for index in indexesiterator:
        yield dataset.get_items(index, pin_memory)


class TensorDataLoader:
    '''DataLoader used to iterate (efficiently!) over tuple of torch tensors'''

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
        if num_workers != 0:
            print("warning: num_workers > 0: num_workers=0 is used instead")
        sampler = RandomSampler if shuffle else SequentialSampler
        self.pin_memory = pin_memory
        self.batch_sampler = BatchSampler(sampler=sampler(
            range(len(dataset))), batch_size=batch_size, drop_last=drop_last)
        self.dataset = dataset

    def __iter__(self):
        return Select_batchnew(self.dataset, self.pin_memory, iter(self.batch_sampler))

    def __len__(self):
        return len(self.batch_sampler)

def select_batch(dataset, pin_memory, indexesiterator, batch_container):
    ''' utility function for TensorDataLoader'''
    for index in indexesiterator:
        batch_index = torch.LongTensor(index)
        res_t=tuple(tensor.index_select(0, batch_index) for tensor in dataset.tensors)
        if batch_container == tuple:
            res = res_t
        else:
            res = batch_container(*res_t)
        if pin_memory and torch.cuda.is_available():
            res = tuple(x.pin_memory() for x in res)
        yield res
    
class TensorDataLoaderOld:
    '''DataLoader used to iterate (efficiently!) over tuple of torch tensors'''

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, batch_container=tuple):
        if num_workers != 0:
            print("warning: num_workers > 0: num_workers=0 is used instead")
        sampler = RandomSampler if shuffle else SequentialSampler
        self.pin_memory = pin_memory
        self.batch_sampler = BatchSampler(sampler=sampler(
            range(len(dataset))), batch_size=batch_size, drop_last=drop_last)
        self.dataset = dataset
        self.batch_container = batch_container

    def __iter__(self):
        return select_batch(self.dataset, self.pin_memory, iter(self.batch_sampler),self.batch_container)

    def __len__(self):
        return len(self.batch_sampler)


def str2bool(x):
    '''convert string to a boolean'''
    if x == 'False' or x == 'True':
        return eval(x)
    else:
        raise Exception(
            'str2bool: cannot convert "{} to a boolean, only "False" or "True" are accepted'.format(x))


class CsvLog:
    '''csv log writer to save all the parameters and metrics during the training'''

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.line = []

    def add2line(self, l):
        # if self.filename is not None:
        self.line += list(map(str, l))

    def writeline(self):
        # if self.filename is not None:
        prefix = "\n"
        if self.file is None:
            self.file = open(self.filename, 'w')
            prefix = ""
        self.file.write(prefix + ",".join(self.line))
        self.file.flush()
        self.line = []

    def close(self):
        if self.file is not None:# and self.filename is not None::
            self.file.close()

if __name__=='__main__':
    import pickle
#    x=torch.Tensor(np.linspace(-10# ,100,100))
    # th = 1
    # print(logexp(x,th)-torch.log(th+torch.exp(x)))
    # print(logexp(x,th))
    # print(torch.log(th+torch.exp(x)))
    # print(np.log(th))
    l=(i for i in range(3))
    m =Infinitecycle(l,10)
    with open("toto.pkl",'wb') as f:
        pickle.dump(m,f)
    # print(len(m))
    # for _ in range(1):
    #     for i in m:
    #         print(i)
