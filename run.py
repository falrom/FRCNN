import argparse
import os

from model import MyEstimator

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--GPU', default='0', help='the GPU No. to use')
subparsers = parser.add_subparsers(dest='mode')

# train mode:
parser_train = subparsers.add_parser(
    'train', help='Train the net. Type "python run.py train -h" for more information.')
parser_train.add_argument('-b', '--B', required=True, type=int, help='the number of recursive blocks')
parser_train.add_argument('-u', '--U', required=True, type=int, help='the number of residual units')
parser_train.add_argument('-c', '--C', required=True, type=int, help='the depth feature maps')
parser_train.add_argument('-v', '--video', required=True,
                          help='video name, like: "BasketballDrive_1920x1080_50_000to049"')
parser_train.add_argument('-q', '--qp', required=True, type=int, help='codex QP')
parser_train.add_argument('-t', '--time_str', help='the old time stamp, from which backup_dir to continue')
parser_train.add_argument('--train_batch', default=200, type=int, help='train batch size (default: 200)')
parser_train.add_argument('--test_batch', default=500, type=int, help='test batch size (default: 500)')
parser_train.add_argument('--steps', type=int,
                          help='how many steps you want to train. If both --steps and --max_steps are None, train for ever.)')
parser_train.add_argument('--max_steps', type=int,
                          help='stop to train when the step reach the max_steps (invalid when --steps is not None)')
parser_train.add_argument('--test_interval', default=50, type=int, help='how often to test (default: 50)')
parser_train.add_argument('--save_interval', default=500, type=int, help='how often to save (default: 200)')
parser_train.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001)')
parser_train.add_argument('--decay', default=0, type=float,
                          help='learning rate decay: lr = lr * (decay^step) (default: No decay)')
parser_train.add_argument('--no_BN_begin', action='store_true', default=False,
                          help='if you do NOT want to use BatchNormalization layer at the beginning of each recursive block')
parser_train.add_argument('--no_BN_ru', action='store_true', default=False,
                          help='if you do NOT want to use BatchNormalization layer in each residual unit')
parser_train.add_argument('--no_BN_end', action='store_true', default=False,
                          help='if you do NOT want to use BatchNormalization layer at the end of the DRRN net')
parser_train.add_argument('--L1', action='store_true', default=False, help='if you want to use the L1 loss to train')

# evaluate mode:
parser_evaluate = subparsers.add_parser(
    'evaluate', help='Evaluate the net. Type "python run.py evaluate -h" for more information.')
parser_evaluate.add_argument('-b', '--B', required=True, type=int, help='the number of recursive blocks')
parser_evaluate.add_argument('-u', '--U', required=True, type=int, help='the number of residual units')
parser_evaluate.add_argument('-c', '--C', required=True, type=int, help='the depth feature maps')
parser_evaluate.add_argument('-v', '--video', required=True,
                             help='video name, like: "BasketballDrive_1920x1080_50_000to049"')
parser_evaluate.add_argument('-q', '--qp', required=True, type=int, help='codex QP')
parser_evaluate.add_argument('--height', required=True, type=int, help='the height of input image/video')
parser_evaluate.add_argument('--width', required=True, type=int, help='the width of input image/video')
parser_evaluate.add_argument('--ckpt', required=True, help='the path of checkpoint')
parser_evaluate.add_argument('-o', '--output', action='store_true', default=False, help='whether to output the results')
parser_evaluate.add_argument('--no_BN_begin', action='store_true', default=False,
                             help='if you do NOT want to use BatchNormalization layer at the beginning of each recursive block')
parser_evaluate.add_argument('--no_BN_ru', action='store_true', default=False,
                             help='if you do NOT want to use BatchNormalization layer in each residual unit')
parser_evaluate.add_argument('--no_BN_end', action='store_true', default=False,
                             help='if you do NOT want to use BatchNormalization layer at the end of the DRRN net')

# parser_predict = subparsers.add_parser(
#     'predict', help='Train the net. Type "python run.py predict -h" for more information.')
# parser_predict.add_argument('--B', required=True, type=int, help='the number of recursive blocks')
# parser_predict.add_argument('--U', required=True, type=int, help='the number of residual units')
# parser_predict.add_argument('--C', required=True, type=int, help='the depth feature maps')
# parser_predict.add_argument('--video', required=True, help='video name, like: "BasketballDrive_1920x1080_50_000to049"')
# parser_predict.add_argument('--qp', required=True, type=int, help='codex QP')
# parser_predict.add_argument('--height', required=True, type=int, help='the height of input image/video')
# parser_predict.add_argument('--width', required=True, type=int, help='the width of input image/video')
# parser_predict.add_argument('--ckpt', required=True, help='the path of checkpoint')
# parser_predict.add_argument('--BN_begin', default=True, type=bool,
#                             help='whether to use BatchNormalization layer at the beginning of each recursive block')
# parser_predict.add_argument('--BN_ru', default=True, type=bool,
#                             help='whether to use BatchNormalization layer in each residual unit')
# parser_predict.add_argument('--BN_end', default=True, type=bool,
#                             help='whether to use BatchNormalization layer at the end of the DRRN net')

args = parser.parse_args()

if args.mode is None:
    parser.print_help()
    exit()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
print('\n** GPU selection:', os.environ["CUDA_VISIBLE_DEVICES"])

estimator = MyEstimator(
    count_B=args.B,
    count_U=args.U,
    channel=args.C,
    QP=args.qp,
    video_name=args.video,
    use_BN_at_begin=(not args.no_BN_begin),
    use_BN_in_ru=(not args.no_BN_ru),
    use_BN_at_end=(not args.no_BN_end)
)

if args.mode == 'train':
    estimator.train(
        time_str=args.time_str,
        train_batch_size=args.train_batch,
        test_batch_size=args.test_batch,
        steps=args.steps,
        max_steps=args.max_steps,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        learning_rate=args.lr,
        decay=args.decay,
        use_L1_loss=args.L1
    )

elif args.mode == 'evaluate':
    estimator.evaluate(
        ckpt_path=args.ckpt,
        height=args.height,
        width=args.width,
        need_output=args.output
    )

# elif args.mode == 'predict':
#     pass
