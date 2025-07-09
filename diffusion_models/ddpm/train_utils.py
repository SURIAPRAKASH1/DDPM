import torch
import torch.optim as optim
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp

from itertools import cycle
import sys, random, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

from diffusion_models.ddpm.gaussian_diffusion import (
    GaussianDiffusion,
    get_linear_beta_schedule, 
    ModelMeanType,
    ModelVarType, 
    LossType
)
from diffusion_models.ddpm.unet import Unet
from comman.argfile import get_args
from default_datasets import PrepareDatasetDataLoader, preprocessing_pipeline

# current device
device = "cuda" if torch.cuda.is_available() else "cpu"

# commant line arguments 
args = get_args()

def setup_dist(rank:int, world_size: int):
    """
    setup distriputed process group. 

    Args:
        rank (int): current process (device) rank 
        world_size (int): How many processes (devices) in a process group
    """

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"  

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend= backend, world_size= world_size, rank= rank)

    
def clean_up():
    "After training we have to destroy DDP"
    dist.destroy_process_group()


def train(rank: int, world_size:int):
    is_ddp = True if world_size > 1 else False

    if is_ddp:
        print("It's a DDP training !")
        print(f"Initializing DDP Rank [{rank}]")
        setup_dist(rank, world_size) 
    else:
        print("It's not a DDP training !")

    # -----------------
    #  DataLoader to handle DDP/Single device
    # -----------------
    if rank == 0 or rank == device: 
        D = PrepareDatasetDataLoader(
            dataset_name= args.dataset_name, 
            transform= preprocessing_pipeline(args.height, args.width), 
            pin_memory = True if device == 'cuda' else False,
            batch_size= args.batch_size, 
            world_size= world_size, 
            rank= rank, 
            shuffle= True, # shuffle is handled correctly in DDP environment
            num_workers= os.cpu_count() if device == 'cuda' else args.num_workers
            )
        dataloader = D.prepare_dataloader()
        
    
        print(f"dataset size = {len(D.dataset)}")
        print("Total Batches =", len(dataloader))
        print(f"world_size = {world_size}")

    #--------------------
    # Initializing Model
    # -------------------

    denoise_model = Unet(
        dim= args.init_dim or args.width, 
        channels= args.channels, 
        out_dim = args.out_dim or args.channels, 
        dim_mults = tuple(args.dim_mults), 
        resnet_block_groups= args.num_groups_in_GN,
        self_condition= args.self_condition, 
        dropout= args.dropout
    ).to(rank)
    torch.compile(denoise_model)

    if is_ddp:
        # wrap model with ddp
        model = DDP(denoise_model, device_ids = rank)
    else:
        model = denoise_model

    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), 
                            lr = args.lr, 
                            betas= tuple(args.adam_betas), 
                            weight_decay= args.weight_decay,
                            fused= args.fused_adam)

    # initiating gaussian diffusion
    gd = GaussianDiffusion(
        betas = get_linear_beta_schedule(args.T) ,
        model_mean_type= ModelMeanType.get_type(args.model_mean_type), 
        model_var_type= ModelVarType.get_type(args.model_var_type), 
        loss_type= LossType.get_type(args.loss_type)
    )
    # ------------------
    # Optimization Loop
    # -----------------

    # infinite dataloader, we can trian model as long as we wanted !.
    infinite_dataloader = cycle(dataloader)
    
    start = time.time()
    for step, (x0s, _) in enumerate(infinite_dataloader):
        # move data to device
        x0s = x0s.to(rank, non_blocking = True if device == 'cuda' else False)
        optimizer.zero_grad() 

        # draw t uniformaly for every sample in a batch
        t = torch.randint(1, args.T, (x0s.size(0), )).long().to(rank)

        # when using bflaot16 we don't need loss scaling
        with torch.autocast(device_type = device, dtype= torch.bfloat16):
            # self condition
            x_self_cond = None
            if args.self_condition and random.random() < 0.5:
                with torch.no_grad():
                    xt = gd.q_sample(x0s, t)
                    pred_xstart =  gd.p_mean_variance(model = model, x = xt, t = t)['pred_xstart']
                    x_self_cond = pred_xstart
                    x_self_cond.detach_()

            # forward-prop & compute loss
            loss = gd.training_losses(model = model, x_start= x0s, t = t, model_kwargs= {"x_self_cond" : x_self_cond})

        # back-prop & update parameters
        loss.backward()
        optimizer.step()

        # save checkpoint for given ckp_interval and final steps
        if (step + 1 % args.ckp_interval == 0 or step == args.steps) and (rank == 0 or rank == device):
            if is_ddp:
                ckp = model.module.state_dict()
            else:
                ckp = model.state_dict() 
            PATH = "ckp.pt"
            torch.save(ckp, PATH)
            print(f"Step {step} | Training checkpoint is saved at {PATH}")

        # printing loss once in a while
        if step % args.print_interval == 0 or step == args.steps:
            print(f"Rank [{rank}] {step}/{args.steps}: loss: {loss.item()}")
        
        # break the training when hit the total training steps
        if step == args.steps:
            break 

    end = time.time()
    print("Training time %.2f" % ((end - start )/60), "M")

    # destroy ddp
    if is_ddp:
        clean_up()

def run():
    world_size = torch.cuda.device_count() if device == 'cuda' else torch.cpu.device_count() # always 1 (not the core)
    if world_size > 1:
        mp.spawn(train, args= (world_size, ), nprocs= world_size, join= True)
    else:
        train(rank = device, world_size)  

if __name__ == "__main__":
    run()

