import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
import argparse
import tensorboard_logger as tb_log

from torch.nn.utils.rnn import pack_padded_sequence
from src import Encoder, get_decoder
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


parser = argparse.ArgumentParser(description="Trian a image caption model")
parser.add_argument("--config", default="coco_mobilenet_rnn.json", help="configuration file", type=str)
args = parser.parse_args()
with open(os.path.join('config', args.config), 'r') as f:
    config = json.load(f)

task_name = args.config.split('.')[0]
print('task name', task_name)
seed = config['seed']
# device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
device = torch.device('cuda')
save_dir = config['save_path']
log_dir = config['log_path']
tb_log.configure(log_dir)

# Data parameters
data_folder = config['data']['data_folder']  # folder with data files saved by create_input_files.py
data_name = config['data']['data_name']  # base name shared by data files

# Model parameters
encoder_type = config['model']['encoder']
encoder_dim = config['model']["encoder_dim"]
emb_dim = config['model']['emb_dim']  # dimension of word embeddings
attention_dim = config['model']['attention_dim']  # dimension of attention linear layers
decoder_dim = config['model']['decoder_dim']  # dimension of decoder RNN
dropout = config['model']['dropout']
decoder_type = config['model']['decoder']

# Training parameters
start_epoch = config['train_config']['start_epoch']
epochs = config['train_config']['epochs']  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = config['train_config']['epochs_since_improvement']  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = config['train_config']['batch_size']
workers = config['train_config']['workers']  # for data-loading; right now, only 1 works with h5py
encoder_lr = config['train_config']['encoder_lr']  # learning rate for encoder if fine-tuning
decoder_lr = config['train_config']['decoder_lr']  # learning rate for decoder
grad_clip = config['train_config']['grad_clip']  # clip gradients at an absolute value of
alpha_c = config['train_config']['alpha_c']  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = config['train_config']['print_freq']  # print training/validation stats every __ batches
fine_tune_encoder = bool(config['train_config']['fine_tune_encoder'])  # fine-tune encoder?

best_bleu4 = 0.  # BLEU-4 score right now
checkpoint = None  # path to checkpoint, None if none

if not os.path.isdir('model_weight'):
    os.mkdir('model_weight')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = get_decoder(
            decoder_type=decoder_type,
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            device=device,
            encoder_dim=encoder_dim,
            dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder(encoder_type=encoder_type)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', ),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, top5accs, losses = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        tb_log.log_value('bleu4', recent_bleu4, epoch)
        tb_log.log_value('test_top5', top5accs, epoch)
        tb_log.log_value('test_loss', losses, epoch)
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0


        filename = os.path.join(save_dir, 'checkpoint_' + str(epoch) + '_' + task_name + '.pth.tar')
        best_file = os.path.join(save_dir, 'BEST_' + task_name + '.pth.tar')
        # Save checkpoint
        save_checkpoint(filename, best_file, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader)):
        data_time.update([time.time() - start], 1)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        scores = scores.to(device)
        targets = targets.to(device)
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean().to(device)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update([time.time() - start], 1)

        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time ({batch_time.avg:.3f})\t'
                  'Data Load Time ({data_time.avg:.3f})\t'
                  'Loss ({loss.avg:.4f})\t'
                  'Top-5 Accuracy ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    tb_log.log_value('train_loss', losses.avg, epoch)
    tb_log.log_value('train-top-5', top5accs.avg, epoch)


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            scores = scores.to(device)
            targets = targets.to(device)
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean().to(device)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update([time.time() - start], 1)

            start = time.time()
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time ({batch_time.avg:.3f})\t'
                      'Loss ({loss.avg:.4f})\t'
                      'Top-5 Accuracy ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, top5accs.avg, losses.avg


if __name__ == '__main__':
    main()
