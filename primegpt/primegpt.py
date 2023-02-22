"""
Trains a character-level language model.
"""

import os
import sys
from livelossplot import PlotLosses

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = PrimeDataSet.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class PrimeDataSet(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, num):
        self.config = config
        self.num = num 
        self.data = self.sieve(self.num)
        chars = [str(s) for s in sorted(list(set(range(10))))]
        data_size, vocab_size = len(self.data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        print(self.stoi)
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size

    def sieve(self, num):
        """
        Compute the nth prime number
        """
        prime = [True for i in range(num+1)]
        # boolean array
        p = 2
        while (p * p <= num):
    
            # If prime[p] is not
            # changed, then it is a prime
            if (prime[p] == True):
    
                # Updating all multiples of p
                for i in range(p * p, num+1, p):
                    prime[i] = False
            p += 1
    
        # Print all prime numbers
        primes = []
        for p in range(2, num+1):
            if prime[p]:
                primes.append(str(p))
        return ''.join(primes) 

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):


        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    liveloss = PlotLosses()
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    
    train_dataset = PrimeDataSet(config.data, 1000)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    logs = {}

    def epoch_end_callack(trainer):
        prefix = ''
        logs[prefix + 'rmse'] = trainer.loss.item()
        liveloss.update(logs)
        liveloss.draw()
    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
             
        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "23571113"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_epoch_end', epoch_end_callack)
    # run the optimization
    trainer.run()
