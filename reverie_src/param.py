import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

      

        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--log_every', type=int, default=2000, help='training iterations')
        self.parser.add_argument('--seed', type=int, default=0, help='training iterations')

        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='vilbert',
                     choices=['vilbert'],
                     help='VLN-BERT backbone')
        self.parser.add_argument('--train', type=str, default='listener',
                    choices=['listener'],
                    help='Retained REVERIE training entrypoint')
        self.parser.add_argument('--description', type=str, default='no description\n')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--maxObject', type=int, default=None, help='Max Object per viewpoint')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument('--directions', type=int, default=4, help='agent-centered visual directions') # fix to 4 for now
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)
        
        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')
        # local-training workflow; the retained FL methods manage their own state.
        # self.parser.add_argument("--loadOptim", action="store_const", default=False, const=True)

        # self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        # self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument('--ref_loss_weight', type=float, default=1.0)
        self.parser.add_argument("--features", type=str, default='img_features/ResNet-152-imagenet.tsv')
        self.parser.add_argument('--init_bert_file', default=None)

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

       

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
    
        self.parser.add_argument('--epsilon', type=float, default=0.1)
        
        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # Federated Learning
        self.parser.add_argument('--fl_mode', type=str, default='c',
                     choices=['c', 'fedavg', 'ours'],
                     help='c: centralized, fedavg: FedAvg, ours: personalized FL')
    
        self.parser.add_argument('--local_epoches', type=float, default=1.0, help='local epochs per client')
        self.parser.add_argument('--sample_fraction', type=float, default=0.2, help='fraction of clients sampled per round')
        self.parser.add_argument('--n_parties', type=int, default=None, help='number of clients (None=all scans)')
        self.parser.add_argument('--disk_n_parties', type=int, default=None,
                    help='number of client states kept on disk in ours mode '
                         '(None=same as n_parties, 0=keep all client states in memory)')
        self.parser.add_argument('--global_lr', type=float, default=1.0, help='global learning rate for aggregation')
        self.parser.add_argument('--comm_round', type=int, default=None, help='total communication rounds (None=auto)')
        
        self.parser.add_argument('--prefix_len', type=int, default=8,
                                help='Number of prefix tokens P per module')
        self.parser.add_argument('--prefix_modules', type=str,
                                default='infer',
                                help='Comma-separated list of layers to add prefix; '
                                     'use "infer" to follow model block order')
        self.parser.add_argument('--gate_hidden', type=int, default=256,
                                help='Hidden dimension of gate network')
       
        # The supported ours method always learns its gate and uses additive prefixes.
        self.parser.add_argument('--attn_prefix_mode', type=str, default='fedperfix_add',
                    choices=['fedperfix_add'],
                    help='Attention prefix mode for the supported additive adapter')
 
        self.parser.add_argument('--prefix_mid_dim', type=int, default=256,
                                help='Bottleneck size for additive attention prefix adapters')
        self.parser.add_argument('--prefix_scale', type=float, default=1.0,
                                help='Global scale for additive attention prefix adapters')
        self.parser.add_argument('--gate_lr', type=float, default=None,
                                help='Learning rate for gate policy (default: lr)')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.01,
                                help='Gate temporal smoothness weight')
        # REMOVE: lambda_budget/lambda_conf are unused gate-regularizer ablations.
        self.parser.add_argument('--prefix_lr', type=float, default=0.00001,
                                help='Learning rate for prefix + gate parameters')
           

        self.parser.set_defaults(
            # The retained ours implementation excludes the language-prefix
            # ablation and always uses visual/biattention prefixes only.
            enable_lang_prefix=False,
            enable_vis_prefix=True,
            enable_bi_prefix=True,
        )

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.description = args.name
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
