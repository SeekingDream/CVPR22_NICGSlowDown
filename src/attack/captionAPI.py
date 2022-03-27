import torch
import torch.nn.functional as F


def prediction(image, encoder, decoder, word_map, beam_size, max_length, device):
    """
    Reads an image and captions it with beam search.
    :param image:
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)

    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()

    complete_seqs_scores = list()
    old_scores_list = list()
    new_scores_list = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    global_incomplete_inds = [i for i in range(k)]
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        now_scores = F.log_softmax(scores, dim=-1)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        new_scores_list = old_scores_list
        for l in range(len(new_scores_list)):
            new_scores_list[l][global_incomplete_inds] = new_scores_list[l][global_incomplete_inds][prev_word_inds]
        old_scores_list = new_scores_list

        next_scores = [now_scores[prev_word_inds[iii], next_word_inds[iii]] for iii in range(len(prev_word_inds))]
        next_scores = torch.stack(next_scores)
        new_scores = torch.zeros([beam_size], device=next_scores.device)
        new_scores[global_incomplete_inds] += next_scores
        old_scores_list.append(new_scores)

        assert (torch.stack(old_scores_list).sum(0)[global_incomplete_inds] - top_k_scores).sum() < 1e-3
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [
            ind for ind, next_word in enumerate(next_word_inds) if
            next_word != word_map['<end>']
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        global_incomplete_inds = [v for i, v in enumerate(global_incomplete_inds) if i not in complete_inds]

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly
        # Proceed with incomplete sequences

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > max_length:
            break
        step += 1
    if len(complete_seqs_scores) == 0:
        complete_seqs_scores = [top_k_scores]
        complete_seqs = seqs.tolist()
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    pred_score = torch.stack(old_scores_list, dim=0)[:, i]
    return seq, pred_score


def prediction_batch(imgs, encoder, decoder, word_map, max_length, device):
    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seq_scores = torch.zeros([batch_size, 1]).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = set()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while len(complete_seqs) < batch_size:
        embeddings = decoder.embedding(k_prev_words.to(device)).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        if type(decoder.decode_step) == torch.nn.LSTMCell:
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        elif type(decoder.decode_step) == torch.nn.GRUCell:
            h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)
        else:
            raise NotImplementedError

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.softmax(scores, dim=-1)
        next_word_probs, next_word_inds = scores.max(1)
        next_word_inds = next_word_inds.cpu()

        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seq_scores = torch.cat([seq_scores, next_word_probs.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = set(range(batch_size)) - set(incomplete_inds)
        complete_seqs.update(complete_inds)
        k_prev_words = next_word_inds.unsqueeze(1)

        if step > max_length:
            break
        step += 1
    k_end_words = torch.LongTensor([[word_map['<end>']]] * batch_size)  # (k, 1)
    seqs = torch.cat([seqs, k_end_words], dim=1)  # (s, step+1)
    k_end_scores = torch.zeros_like(k_end_words).to(device)
    seq_scores = torch.cat([seq_scores, k_end_scores], dim=1)
    # seq_length = [s.tolist().index(word_map['<end>']) for s in seqs]
    return seqs, seq_scores


def prediction_batch_end(imgs, encoder, decoder, word_map, max_length, device):
    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seq_scores = torch.zeros([batch_size, 1]).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = set()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while len(complete_seqs) < batch_size:
        embeddings = decoder.embedding(k_prev_words.to(device)).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.softmax(scores, dim=-1)
        _, next_word_inds = scores.max(1)
        next_word_inds = next_word_inds.cpu()
        next_word_probs = scores[:, word_map['<end>']]

        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seq_scores = torch.cat([seq_scores, next_word_probs.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = set(range(batch_size)) - set(incomplete_inds)
        complete_seqs.update(complete_inds)
        k_prev_words = next_word_inds.unsqueeze(1)

        if step > max_length:
            break
        step += 1
    k_end_words = torch.LongTensor([[word_map['<end>']]] * batch_size)  # (k, 1)
    seqs = torch.cat([seqs, k_end_words], dim=1)  # (s, step+1)
    k_end_scores = torch.zeros_like(k_end_words).to(device)
    seq_scores = torch.cat([seq_scores, k_end_scores], dim=1)
    # seq_length = [s.tolist().index(word_map['<end>']) for s in seqs]
    return seqs, seq_scores


def prediction_batch_target_end(imgs, encoder, decoder, word_map, max_length, device):
    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seq_scores = torch.zeros([batch_size, 1]).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = set()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while len(complete_seqs) < batch_size:
        embeddings = decoder.embedding(k_prev_words.to(device)).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        if type(decoder.decode_step) == torch.nn.LSTMCell:
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        elif type(decoder.decode_step) == torch.nn.GRUCell:
            h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)
        else:
            raise NotImplementedError

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.softmax(scores, dim=-1)
        next_word_probs, next_word_inds = scores.max(1)

        next_word_probs = \
            scores[:, word_map['<end>']] + \
            (next_word_inds != word_map['<end>']) * next_word_probs

        next_word_inds = next_word_inds.cpu()
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seq_scores = torch.cat([seq_scores, next_word_probs.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = set(range(batch_size)) - set(incomplete_inds)
        complete_seqs.update(complete_inds)
        k_prev_words = next_word_inds.unsqueeze(1)

        if step > max_length:
            break
        step += 1
    k_end_words = torch.LongTensor([[word_map['<end>']]] * batch_size)  # (k, 1)
    seqs = torch.cat([seqs, k_end_words], dim=1)  # (s, step+1)
    k_end_scores = torch.zeros_like(k_end_words).to(device)
    seq_scores = torch.cat([seq_scores, k_end_scores], dim=1)
    # seq_length = [s.tolist().index(word_map['<end>']) for s in seqs]
    return seqs, seq_scores


def get_seq_len(seq, word_map):
    for i, tk in enumerate(seq):
        if tk == word_map['<end>']:
            return i
    return len(seq)


@torch.no_grad()
def prediction_len_batch(imgs, encoder, decoder, word_map, max_length, device):
    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seq_scores = torch.zeros([batch_size, 1]).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = set()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while len(complete_seqs) < batch_size:
        embeddings = decoder.embedding(k_prev_words.to(device)).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        if type(decoder.decode_step) == torch.nn.LSTMCell:
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        elif type(decoder.decode_step) == torch.nn.GRUCell:
            h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)
        else:
            raise NotImplementedError

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=-1)
        _, next_word_inds = scores.max(1)
        next_word_inds = next_word_inds.cpu()

        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = set(range(batch_size)) - set(incomplete_inds)
        complete_seqs.update(complete_inds)
        k_prev_words = next_word_inds.unsqueeze(1)

        if step > max_length:
            break
        step += 1
    k_end_words = torch.LongTensor([[word_map['<end>']]] * batch_size)  # (k, 1)
    seqs = torch.cat([seqs, k_end_words], dim=1)  # (s, step+1)
    return [get_seq_len(seq, word_map) for seq in seqs]



